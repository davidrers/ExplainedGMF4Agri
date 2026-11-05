"""The PaPs loss, vendored from https://github.com/VSainteuf/utae-paps (commit 987874e, MIT License, Copyright (c)
2021 VSainteuf): ``src/panoptic/paps_loss.py`` (``PaPsLoss``, ``CenterLoss``) and ``src/panoptic/FocalLoss.py``.

The loss is the source's: the centerness loss of CornerNet on the heatmap (alpha 0, beta 4), and, on the centre kept
for each true parcel, the normalised L1 of its box size, the cross-entropy of its class (focal with gamma 0) and the
binary cross-entropy of its mask, all weighted 1. What changes, and why:

* **Land without supervision.** PASTIS labels every pixel. Under a label budget only the drawn parcels are known, and
  WorldCover's non-agricultural land is the only background (``targets.py``); every other pixel is unknown. The
  target carries a ``known`` channel, and the centerness loss is summed over the known pixels only. It is still
  divided by the number of centres, as in the source.
* **Centres outside every zone.** Zone 0 is unknown land. The class, size and shape losses use only the centres that
  lie in a zone > 0, which in training (``supmax``) are all of them; at validation, where every local maximum is a
  centre, the others would otherwise be supervised with a class and a size they do not have.
* **No centre matched.** The source concatenates the shape predictions of the matched centres and fails when there
  are none; the shape loss is then 0 here.

The target layout is the source's ``(B, H, W, 7)`` (heatmap, instance id, zone, height, width, object class, pixel
class) with the ``known`` channel appended as an eighth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """The source's focal loss (from clcarwin/focal_loss_pytorch) with an ignored label; gamma 0 is cross-entropy."""

    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_label=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1).transpose(1, 2).contiguous().view(-1, input.size(1))
        target = target.view(-1, 1)
        if input.squeeze(1).dim() == 1:
            logpt = torch.sigmoid(input).view(-1)
        else:
            logpt = F.log_softmax(input, dim=-1).gather(1, target).view(-1)
        pt = logpt.detach().exp()
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            logpt = logpt * self.alpha.gather(0, target.data.view(-1))
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.ignore_label is not None:
            loss = loss[target[:, 0] != self.ignore_label]
        return loss.mean() if self.size_average else loss.sum()


class CenterLoss(nn.Module):
    """CornerNet's centerness loss (``kp_utils.py``), as the source adapts it, over the known pixels."""

    def __init__(self, alpha=0, beta=4, eps=1e-8):
        super().__init__()
        self.a, self.b, self.eps = alpha, beta, eps

    def forward(self, preds, gt, known=None):
        pred = preds.permute(0, 2, 3, 1).contiguous().view(-1, preds.shape[1])
        g = gt.view(-1, preds.shape[1])
        k = torch.ones_like(g, dtype=torch.bool) if known is None else known.reshape(-1, 1).bool()
        pos_inds = g.eq(1) & k
        neg_inds = g.lt(1) & k
        num_pos = pos_inds.float().sum()
        pos_loss = (torch.log(pred[pos_inds] + self.eps) * torch.pow(1 - pred[pos_inds], self.a)).sum()
        neg_loss = torch.log(1 - pred[neg_inds] + self.eps) * torch.pow(pred[neg_inds], self.a)
        neg_loss = (neg_loss * torch.pow(1 - g[neg_inds], self.b)).sum()
        if pred[pos_inds].nelement() == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


class PaPsLoss(nn.Module):
    def __init__(self, l_center=1, l_size=1, l_shape=1, l_class=1, alpha=0, beta=4, gamma=0, eps=1e-8,
                 void_label=None, binary_threshold=0.4):
        super().__init__()
        self.l_center, self.l_size, self.l_shape, self.l_class = l_center, l_size, l_shape, l_class
        self.eps = eps
        self.binary_threshold = binary_threshold
        self.center_loss = CenterLoss(alpha=alpha, beta=beta, eps=eps)
        self.class_loss = FocalLoss(gamma=gamma, ignore_label=void_label)
        self.shape_loss = FocalLoss(gamma=0)
        self.value = (0, 0, 0, 0)
        # The confidences of the centres matched to a true parcel and the IoU of their masks, for tuning the
        # pseudo-NMS confidence threshold on the validation chips (the source's ``autotune``).
        self.predicted_confidences = None
        self.achieved_ious = None

    def forward(self, predictions, target, heatmap_only=False):
        target_heatmap, true_instances, zones, size, sem_obj, sem_pix, known = target.split((1, 1, 1, 2, 1, 1, 1),
                                                                                          dim=-1)
        loss_center = loss_size = loss_shape = loss_class = 0
        if self.l_center != 0:
            loss_center = self.center_loss(predictions["heatmap"], target_heatmap, known)

        cm = predictions["center_mask"]
        self.predicted_confidences = self.achieved_ious = None
        if not heatmap_only and predictions["size"] is not None and predictions["size"].shape[0] != 0:
            # Centres in a zone > 0 only (module docstring); every centre's row in the head outputs.
            in_zone = zones.squeeze(-1)[cm] > 0
            if in_zone.any():
                if self.l_size != 0:
                    true_size = size[cm][in_zone]
                    ls = torch.abs(true_size - predictions["size"][in_zone]) / (true_size + self.eps)
                    loss_size = ls.sum(dim=-1).mean()
                if self.l_class != 0:
                    loss_class = self.class_loss(predictions["semantic"][in_zone], sem_obj[cm][in_zone].long())

            if self.l_shape != 0 and in_zone.any():
                center_mapping = {(int(b), int(i), int(j)): k for k, (b, i, j) in enumerate(zip(*torch.where(cm)))}
                confidence_pred, ious, flatten_preds, flatten_target = [], [], [], []
                for b, instance_mask in enumerate(true_instances):
                    zb = zones[b].squeeze(-1)
                    for inst_id in torch.unique(instance_mask):
                        if inst_id == 0:
                            continue
                        centers = cm[b] * (zb == inst_id)
                        if not centers.any():
                            continue
                        for x, y in centers.nonzero():
                            true_mask = (instance_mask == inst_id).squeeze(-1).float()
                            pred_id = center_mapping[(b, int(x), int(y))]
                            xtl, ytl, xbr, ybr = predictions["instance_boxes"][pred_id]
                            crop_true = true_mask[ytl:ybr, xtl:xbr].reshape(-1, 1)
                            mask = predictions["instance_masks"][pred_id].reshape(-1, 1)
                            flatten_preds.append(mask)
                            flatten_target.append(crop_true)
                            confidence_pred.append(predictions["confidence"][pred_id])
                            bmask = mask > self.binary_threshold
                            inter = (bmask * crop_true).sum().float()
                            union = ((bmask + crop_true) != 0).sum()
                            true_mask[ytl:ybr, xtl:xbr] = 0
                            union = union + true_mask.sum()
                            iou = inter / union
                            if torch.isnan(iou) or torch.isinf(iou):
                                iou = torch.zeros_like(iou)
                            ious.append(iou)
                if flatten_preds:
                    p = torch.cat(flatten_preds, dim=0)
                    p = torch.cat([1 - p, p], dim=1)
                    t = torch.cat(flatten_target, dim=0).long()
                    # The source passes probabilities where its focal loss expects scores; kept as it is.
                    loss_shape = self.shape_loss(p, t)
                    self.predicted_confidences = torch.stack(confidence_pred)
                    self.achieved_ious = torch.stack(ious).unsqueeze(-1)

        loss = (self.l_center * loss_center + self.l_size * loss_size + self.l_shape * loss_shape
                + self.l_class * loss_class)
        self.value = tuple(float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                           for v in (loss_center, loss_size, loss_shape, loss_class))
        return loss
