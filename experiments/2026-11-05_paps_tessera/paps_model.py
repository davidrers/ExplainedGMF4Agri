"""U-Net + PaPs on a 10 m embedding raster: the Parcels-as-Points module of utae-paps on U-TAE's spatial path.

Vendored from https://github.com/VSainteuf/utae-paps (commit 987874e, MIT License, Copyright (c) 2021 VSainteuf):
``src/panoptic/paps.py`` (``PaPs``, ``CenterExtractor``), ``src/backbones/utae.py`` (``ConvLayer``, ``ConvBlock``,
``DownConvBlock``, ``UpConvBlock``) and ``src/learning/weight_init.py``. What changes, and why:

* **The backbone.** U-TAE encodes a time series: its convolutions run on every date, an L-TAE attention pools the
  dates at the lowest resolution, and its attention masks pool the skips. TESSERA has already pooled the time series
  of each pixel into 128 numbers, so :class:`UNet` keeps U-TAE's spatial encoder and decoder (the same widths,
  ``[64, 64, 64, 128]`` down, ``[32, 32, 64, 128]`` up, strided 4/2/1 convolutions, group norm in the encoder and
  batch norm in the decoder, reflect padding) and drops the temporal parts: each skip is the encoder's own map.
  It returns, as U-TAE does for PaPs, the full-resolution 32-channel map and the decoder's maps from 1/8 to 1/1.
* **Centres during training.** ``CenterExtractor`` takes, per zone, the highest local value of the heatmap
  (``supmax``). The source does this with ``torch_scatter.scatter_max``; this copy uses ``torch.scatter_reduce``,
  the same result without the dependency. Zone 0 here is land without supervision (``targets.py``) and gets no
  centre.
* **Batches of one.** ``center_mask.squeeze()`` and ``confidence[...].squeeze()`` collapse a batch of one or a
  single candidate; they are ``squeeze(1)`` and ``view(-1)`` here, so the last partial test batch is predicted
  too (the source drops it).
* **Pruning before the masks.** At inference, candidates below ``min_confidence`` never enter the pseudo-NMS, so
  their masks are not assembled (``prune``); the panoptic output is the same, and it is much faster.

Nothing else is changed: the heatmap, saliency, shape, size and class heads, the mask assembly and the pseudo-NMS
are the source's, with its defaults (shape 16 x 16, mask CNN, confidence 0.2 until tuned, ``min_remain`` 0.5, mask
threshold 0.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# --------------------------------------------------------------------------------- U-TAE's spatial blocks


class ConvLayer(nn.Module):
    def __init__(self, nkernels, norm="batch", k=3, s=1, p=1, n_groups=4, last_relu=True, padding_mode="reflect"):
        super().__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            def nl(num_feats):
                return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(nn.Conv2d(in_channels=nkernels[i], out_channels=nkernels[i + 1], kernel_size=k,
                                    padding=p, stride=s, padding_mode=padding_mode))
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(nn.Module):
    def __init__(self, nkernels, norm="batch", last_relu=True, padding_mode="reflect"):
        super().__init__()
        self.conv = ConvLayer(nkernels=nkernels, norm=norm, last_relu=last_relu, padding_mode=padding_mode)

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(nn.Module):
    def __init__(self, d_in, d_out, k, s, p, norm="batch", padding_mode="reflect"):
        super().__init__()
        self.down = ConvLayer(nkernels=[d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode)
        self.conv1 = ConvLayer(nkernels=[d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
                                       nn.BatchNorm2d(d), nn.ReLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s,
                                                   padding=p),
                                nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)


class UNet(nn.Module):
    """U-TAE's spatial encoder and decoder without its temporal attention, on a ``(B, C, H, W)`` raster.

    Returns ``(out, maps)``: the full-resolution decoder map (``decoder_widths[0]`` channels) and the decoder's maps
    from the lowest resolution to the highest, as U-TAE with ``encoder=True`` returns them for PaPs.
    """

    def __init__(self, input_dim, encoder_widths=(64, 64, 64, 128), decoder_widths=(32, 32, 64, 128),
                 str_conv_k=4, str_conv_s=2, str_conv_p=1, encoder_norm="group", padding_mode="reflect"):
        super().__init__()
        encoder_widths, decoder_widths = list(encoder_widths), list(decoder_widths)
        assert len(encoder_widths) == len(decoder_widths) and encoder_widths[-1] == decoder_widths[-1]
        self.n_stages = len(encoder_widths)
        self.enc_dim = decoder_widths[0]
        self.stack_dim = sum(decoder_widths)
        self.in_conv = ConvBlock(nkernels=[input_dim, encoder_widths[0], encoder_widths[0]], norm=encoder_norm,
                                 padding_mode=padding_mode)
        self.down_blocks = nn.ModuleList(
            DownConvBlock(d_in=encoder_widths[i], d_out=encoder_widths[i + 1], k=str_conv_k, s=str_conv_s,
                          p=str_conv_p, norm=encoder_norm, padding_mode=padding_mode)
            for i in range(self.n_stages - 1))
        self.up_blocks = nn.ModuleList(
            UpConvBlock(d_in=decoder_widths[i], d_out=decoder_widths[i - 1], d_skip=encoder_widths[i - 1],
                        k=str_conv_k, s=str_conv_s, p=str_conv_p, norm="batch", padding_mode=padding_mode)
            for i in range(self.n_stages - 1, 0, -1))

    def forward(self, input, batch_positions=None):
        feature_maps = [self.in_conv(input)]
        for i in range(self.n_stages - 1):
            feature_maps.append(self.down_blocks[i](feature_maps[-1]))
        out = feature_maps[-1]
        maps = [out]
        for i in range(self.n_stages - 1):
            out = self.up_blocks[i](out, feature_maps[-(i + 2)])
            maps.append(out)
        return out, maps


def weight_init(m):
    """utae-paps' ``weight_init`` for the layer types used here."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        init.normal_(m.weight.data, mean=0, std=1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)


# -------------------------------------------------------------------------------------------------- PaPs


class PaPs(nn.Module):
    def __init__(self, encoder, num_classes=20, shape_size=16, mask_conv=True, min_confidence=0.2, min_remain=0.5,
                 mask_threshold=0.4):
        super().__init__()
        self.encoder = encoder
        self.shape_size = shape_size
        self.num_classes = num_classes
        self.min_scale = 1 / shape_size
        self.register_buffer("min_confidence", torch.tensor([min_confidence]))
        self.min_remain = min_remain
        self.mask_threshold = mask_threshold
        self.center_extractor = CenterExtractor()

        enc_dim = encoder.enc_dim
        stack_dim = encoder.stack_dim
        self.heatmap_conv = nn.Sequential(
            ConvLayer(nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1, padding_mode="reflect"), nn.Sigmoid())
        self.saliency_conv = ConvLayer(nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1, padding_mode="reflect")
        self.shape_mlp = nn.Sequential(nn.Linear(stack_dim, stack_dim // 2), nn.BatchNorm1d(stack_dim // 2),
                                       nn.ReLU(), nn.Linear(stack_dim // 2, shape_size ** 2))
        self.size_mlp = nn.Sequential(nn.Linear(stack_dim, stack_dim // 2), nn.BatchNorm1d(stack_dim // 2),
                                      nn.ReLU(), nn.Linear(stack_dim // 2, stack_dim // 4),
                                      nn.BatchNorm1d(stack_dim // 4), nn.ReLU(), nn.Linear(stack_dim // 4, 2),
                                      nn.Softplus())
        self.class_mlp = nn.Sequential(nn.Linear(stack_dim, stack_dim // 2), nn.BatchNorm1d(stack_dim // 2),
                                       nn.ReLU(), nn.Linear(stack_dim // 2, stack_dim // 4),
                                       nn.Linear(stack_dim // 4, num_classes))
        if mask_conv:
            self.mask_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.GroupNorm(num_channels=16, num_groups=1), nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1))
        else:
            self.mask_cnn = None

    def forward(self, input, batch_positions=None, zones=None, pseudo_nms=True, heatmap_only=False, prune=False):
        """As the source's ``PaPs.forward``; ``prune`` drops candidates below ``min_confidence`` before their
        masks are assembled (inference only: the pseudo-NMS never keeps them)."""
        out, maps = self.encoder(input, batch_positions=batch_positions)

        heatmap = self.heatmap_conv(out)
        saliency = self.saliency_conv(out)

        center_mask, _ = self.center_extractor(heatmap, zones=zones)
        center_mask = center_mask.squeeze(1)                     # (B, H, W), also for B = 1
        if prune:
            center_mask = center_mask & (heatmap.squeeze(1) >= self.min_confidence)

        if heatmap_only:
            return dict(center_mask=center_mask, saliency=None, heatmap=heatmap, semantic=None, size=None,
                        offsets=None, confidence=None, instance_masks=None, instance_boxes=None,
                        pano_instance=None, pano_semantic=None)

        H, W = heatmap.shape[-2:]
        center_batch, center_h, center_w = torch.where(center_mask)
        center_positions = torch.stack([center_h, center_w], dim=1)

        stack = []
        for i, m in enumerate(maps):
            h_mask = center_h // (2 ** (len(maps) - 1 - i))
            w_mask = center_w // (2 ** (len(maps) - 1 - i))
            m = m.permute(0, 2, 3, 1)
            stack.append(m[center_batch, h_mask, w_mask])
        stack = torch.cat(stack, dim=1)

        if stack.shape[0] == 1 and self.training:
            # BatchNorm1d cannot train on one centre; the step is supervised by the heatmap alone.
            stack = stack.repeat(2, 1)
            size, sem, shapes = self.size_mlp(stack)[:1], self.class_mlp(stack)[:1], self.shape_mlp(stack)[:1]
        else:
            size = self.size_mlp(stack)
            sem = self.class_mlp(stack)
            shapes = self.shape_mlp(stack)
        shapes = shapes.view((-1, 1, self.shape_size, self.shape_size))

        centerness = heatmap[center_mask[:, None, :, :]].unsqueeze(-1)
        confidence = centerness

        clamp_size = size.detach().round().long().clamp_min(min=1)
        half_size = clamp_size // 2
        remainder_size = clamp_size % 2
        start_hw = center_positions - half_size
        stop_hw = center_positions + half_size + remainder_size

        instance_boxes = torch.cat([start_hw, stop_hw], dim=1)
        instance_boxes.clamp_(min=0, max=H)
        instance_boxes = instance_boxes[:, [1, 0, 3, 2]]

        valid_start = (-start_hw).clamp_(min=0)
        valid_stop = (stop_hw - start_hw) - (stop_hw - H).clamp_(min=0)

        instance_masks = []
        for i, s in enumerate(shapes.split(1, dim=0)):
            h, w = clamp_size[i]
            w_start, h_start, w_stop, h_stop = instance_boxes[i]
            h_start_valid, w_start_valid = valid_start[i]
            h_stop_valid, w_stop_valid = valid_stop[i]
            pred_mask = F.interpolate(s, size=(h.item(), w.item()), mode="bilinear").squeeze(0)
            pred_mask = pred_mask[:, h_start_valid:h_stop_valid, w_start_valid:w_stop_valid]
            crop_saliency = saliency[center_batch[i], :, h_start:h_stop, w_start:w_stop]
            if self.mask_cnn is None:
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(crop_saliency)
            else:
                pred_mask = pred_mask + crop_saliency
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(self.mask_cnn(pred_mask.unsqueeze(0)).squeeze(0))
            instance_masks.append(pred_mask)

        if pseudo_nms:
            panoptic_instance, panoptic_semantic = [], []
            for b in range(center_mask.shape[0]):
                panoptic_mask = torch.zeros(center_mask[0].shape, device=center_mask.device)
                semantic_mask = torch.zeros((self.num_classes, *center_mask[0].shape), device=center_mask.device)
                candidates = torch.where(center_batch == b)[0]
                for n, (c, idx) in enumerate(zip(*torch.sort(confidence[candidates].view(-1), descending=True))):
                    if c < self.min_confidence:
                        break
                    new_mask = torch.zeros(center_mask[0].shape, device=center_mask.device)
                    pred_mask_bin = (instance_masks[candidates[idx]].squeeze(0) > self.mask_threshold).float()
                    if pred_mask_bin.sum() > 0:
                        xtl, ytl, xbr, ybr = instance_boxes[candidates[idx]]
                        new_mask[ytl:ybr, xtl:xbr] = pred_mask_bin
                        if ((new_mask != 0) * (panoptic_mask != 0)).any():
                            n_total = (new_mask != 0).sum()
                            non_overlaping_mask = (new_mask != 0) * (panoptic_mask == 0)
                            n_new = non_overlaping_mask.sum().float()
                            if n_new / n_total > self.min_remain:
                                panoptic_mask[non_overlaping_mask] = n + 1
                                semantic_mask[:, non_overlaping_mask] = sem[candidates[idx]][:, None]
                        else:
                            panoptic_mask[(new_mask != 0)] = n + 1
                            semantic_mask[:, (new_mask != 0)] = sem[candidates[idx]][:, None]
                panoptic_instance.append(panoptic_mask)
                panoptic_semantic.append(semantic_mask)
            panoptic_instance = torch.stack(panoptic_instance, dim=0)
            panoptic_semantic = torch.stack(panoptic_semantic, dim=0)
        else:
            panoptic_instance = panoptic_semantic = None

        return dict(center_mask=center_mask, saliency=saliency, heatmap=heatmap, semantic=sem, size=size,
                    confidence=confidence, centerness=centerness, instance_masks=instance_masks,
                    instance_boxes=instance_boxes, pano_instance=panoptic_instance, pano_semantic=panoptic_semantic)


class CenterExtractor(nn.Module):
    """Local maxima of the heatmap; with ``zones``, the highest pixel of each zone > 0 instead."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, input, zones=None):
        if zones is not None:
            masks = []
            for b, x in enumerate(input.split(1, dim=0)):
                x = x.reshape(-1)
                z = zones[b].reshape(-1).long()
                n = int(z.max().item()) + 1
                zmax = torch.full((n,), -1.0, device=x.device, dtype=x.dtype).scatter_reduce(
                    0, z, x, reduce="amax", include_self=True)
                pos = torch.arange(x.numel(), device=x.device)
                hit = (x == zmax[z]) & (z > 0)
                first = torch.full((n,), x.numel(), device=x.device, dtype=torch.long).scatter_reduce(
                    0, z[hit], pos[hit], reduce="amin", include_self=True)
                mask = torch.zeros(x.shape, device=x.device)
                mask[first[first < x.numel()]] = 1
                masks.append(mask.view(zones[b].shape[:2]).unsqueeze(0))
            centermask = torch.stack(masks, dim=0).bool()
        else:
            centermask = input == self.pool(input)
            no_valley = input > input.mean()
            centermask = centermask * no_valley
        n_centers = int(centermask.sum().detach().cpu())
        return centermask, n_centers


def build_model(input_dim: int, num_classes: int, cfg: dict | None = None) -> PaPs:
    """U-Net + PaPs with utae-paps' defaults, initialised with its ``weight_init``."""
    cfg = cfg or {}
    enc = UNet(input_dim, encoder_widths=cfg.get("encoder_widths", (64, 64, 64, 128)),
               decoder_widths=cfg.get("decoder_widths", (32, 32, 64, 128)))
    model = PaPs(enc, num_classes=num_classes, shape_size=cfg.get("shape_size", 16),
                 mask_conv=cfg.get("mask_conv", True), min_confidence=cfg.get("min_confidence", 0.2),
                 min_remain=cfg.get("min_remain", 0.5), mask_threshold=cfg.get("mask_threshold", 0.4))
    model.apply(weight_init)
    return model
