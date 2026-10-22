"""Build index.html: the Phase 1 pipeline page in the style of the Claude Design file
"Architecture Diagram v2" (project cbea8928), with step-through, lane focus, live flow and zoom.

    python docs/phase1/pipeline_page/build_page.py

Section 1 recreates the design's experimental workflow, section 2 its model architecture with the
current arms, section 3 lists the six experiments of the analysis with links to their reports.
Coordinates of section 2 are in design pixels; the page scales each diagram to the window.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

OUT = Path(__file__).with_name("index.html")
REPO = "https://github.com/davidrers/ExplainedGMF4Agri/blob/main"
E = html.escape

# ----------------------------------------------------------------------------------------------
# Section 1: experimental workflow (the design's flex layout, its colours and sizes)
# ----------------------------------------------------------------------------------------------

BLUE = dict(circle="#3f6db0", bg="#e9f0fa", border="#4a74b5")
GREY = dict(circle="#4a5160", bg="#f6f6f4", border="#a8a8a8")
BROWN = dict(circle="#8a5a14", bg="#fcf6ec", border="#b9862a")
GREEN = dict(circle="#2e6b3a", bg="#e6f0e8", border="#3d8a4f")

STATUS = {"met": ("Built", "st-met"), "partly": ("Partly", "st-part"), "no": ("Not yet", "st-no")}


def inner(title, text, bg="#fff", border="#a9c1e6"):
    return (f'<div style="flex:1;box-sizing:border-box;background:{bg};border:2px solid {border};border-radius:6px;'
            f'padding:20px 20px;min-height:92px"><div style="font-size:19px;font-weight:600;margin-bottom:4px">{title}</div>'
            f'<div style="font-size:19px;color:#2a2f38">{text}</div></div>')


ARROW_H = ('<div class="conn-h" style="flex:none;width:64px;display:flex;align-items:center"><div class="shaft-h">'
           '<span class="clip"><i></i></span></div><div style="border-top:5px solid transparent;border-bottom:5px solid '
           'transparent;border-left:9px solid #5a6270"></div></div>')

WF_STEPS = [
    dict(n=1, c=BLUE, title="Zone screening", tags=[], status="no",
         body="EuroCrops countries mapped to GAEZ v5 agro-ecological zones. Transfer only within the same climate zone.",
         now="Not built yet: only Estonia 2021 has run."),
    dict(n=2, c=dict(BLUE, bg="#f1f5fb"), title="Data extraction", tags=["F1"], status="met", boxed=True,
         body=('<div style="display:flex;gap:32px;margin:0 -6px">'
               + inner("TerraMind · THOR · Prithvi · TESSERA · U-Net", "Sentinel-2 + Sentinel-1, 12 monthly composites")
               + inner("AlphaEarth", "Precomputed embeddings, raw extraction")
               + '</div><div style="margin:20px -6px 0;background:#dce6f4;border:2px solid #4a74b5;border-radius:6px;'
               'padding:14px;text-align:center;font-size:19px;font-weight:600">Same chips and parcel label masks</div>'),
         walk="Every model reads the same chips and parcel label masks. Five arms read the monthly Sentinel-2 + "
              "Sentinel-1 composites; AlphaEarth its precomputed embedding.",
         now="EE_2021_rev: 7,402 chips of 224 × 224 px at 10 m, 20 classes. TESSERA v2 runs on the composites; "
             "Prithvi reads Sentinel-2 only."),
    dict(n=3, c=BLUE, title="Spatial split", tags=["F3", "F5"], status="met",
         body="Train, validation and test by whole blocks, with a buffer. Test blocks fixed and fully labelled.",
         now="Blocks of 4 × 4 chips with a 1,600 m buffer: 4,898 training, 769 validation and 1,475 test chips."),
    dict(n=4, c=BLUE, title="Label budget", tags=["F2", "F7"], status="partly",
         body="K = 1, 5, 10, 20, 50, 100 % of parcels per class. Nested, repeated draws.",
         now="Run so far: K = 100, 20 and 5 %, one draw. Draws are not yet nested across K."),
    dict(n=5, c=GREY, title="Models and training", tags=["F4", "F6"], status="partly", boxed=True,
         body=('<div style="display:flex;align-items:center;margin:0 -6px">'
               + inner("Foundation model", "frozen", "#ececec", "#666") + ARROW_H
               + inner("Decoder", "trained", "#fbefd9", "#c08a2a")
               + '<div style="flex:none;width:32px"></div>'
               + inner("Baseline: U-Net", "trained from scratch", "#fbefd9", "#c08a2a")
               + '</div><div style="margin:20px -6px 0;border:2px dashed #666;border-radius:6px;padding:14px;'
               'text-align:center;font-size:19px;font-weight:600">Same training effort: equal tuning trials, same number '
               'of training steps at every K</div>'),
         walk="A frozen foundation model feeds a trained decoder; the U-Net baseline trains from scratch. Every arm "
              "gets the same training effort at every K.",
         now="Six arms, 45 epochs each at every K; the checkpoint of lowest validation loss is tested. "
             "No tuning trials yet."),
    dict(n=6, c=BROWN, title="Experiments", tags=["F2", "F3"], status="partly", boxed=True,
         body=('<div style="display:flex;gap:32px;margin:0 -6px">'
               + inner("E1 · Label-budget curves", "within each country", "#fff", "#e2b56a")
               + inner("E2 · Cross-country transfer, same zone", "zero-shot · target only · source + target", "#fff",
                       "#e2b56a") + '</div>'),
         walk="E1: label-budget curves within each country. E2: cross-country transfer within one agro-ecological "
              "zone, zero-shot, target only, and source plus target.",
         now="E1 has run on Estonia, one draw and one seed per cell. E2 has not started."),
    dict(n=7, c=GREEN, title="Evaluation", tags=["F3", "F7", "F8"], status="partly",
         body="Fixed test blocks only. Macro-F1 (primary) per pixel and per parcel, mIoU and per-class F1, also per AEZ "
              "class.<br>Paired comparisons between models, with 95 % intervals.",
         now="Pixel and parcel Macro-F1 on the 1,475 test chips. No intervals yet: one draw, one seed."),
    dict(n=8, c=GREEN, title="Reporting", tags=["F8"], status="met",
         body="Every result logged with its settings; all draws released.",
         now="Every fit writes results.json with its settings, drawn parcels, metrics and commit."),
]

RULES = [
    ("F1", "Same input", "Identical chips and masks for every model.", "met", "One chip set, EE_2021_rev."),
    ("F2", "Same labels", "Identical parcels at each K and draw.", "met",
     "The draw depends on class, K and draw only."),
    ("F3", "Same test", "One fixed, spatially separated test set.", "met", "1,475 test chips for every fit."),
    ("F4", "Same effort", "Equal tuning trials and the same number of training steps for every model and K.", "partly",
     "45 epochs for every arm and K; no tuning trials yet."),
    ("F5", "No leakage", "Buffer around held-out blocks; no validation labels at low K.", "partly",
     "1,600 m buffer; validation labels still choose the checkpoint at every K."),
    ("F6", "Like with like", "Headline comparisons within the same resolution group.", "partly",
     "Groups declared; the U-Net baseline belongs to neither."),
    ("F7", "Repeated and paired", "Several draws per K; models compared on the same draws.", "no",
     "Paired on one draw and one seed so far."),
    ("F8", "Fixed in advance", "Metrics and criteria declared before results are seen.", "partly",
     "Macro-F1 fixed; the checkpoint criterion is still open."),
]


def status_chip(key):
    label, cls = STATUS[key]
    return f'<span class="st {cls}">{label}</span>'


def wf_step(s):
    c = s["c"]
    pill = ""
    if s["tags"]:
        pill = ('<div class="fpill">' + " ".join(f'<span data-f="{t}">{t}</span>' for t in s["tags"]) + "</div>")
    if s.get("boxed"):
        head = f'<div style="font-size:24px;font-weight:600;margin:0 0 18px -6px">{s["title"]}{status_chip(s["status"])}</div>'
        body = s["body"]
    else:
        head = f'<div style="font-size:25px;font-weight:600;margin-bottom:6px">{s["title"]}{status_chip(s["status"])}</div>'
        body = f'<div style="font-size:19px;line-height:1.5;color:#2a2f38">{s["body"]}</div>'
    tags = " ".join(s["tags"])
    return (f'<div class="wstep n" id="ws{s["n"]}" data-tags="{tags}" style="display:flex;gap:20px;align-items:flex-start">'
            f'<div style="flex:none;width:46px;height:46px;border-radius:50%;background:{c["circle"]};color:#fff;display:flex;'
            f'align-items:center;justify-content:center;font-weight:700;font-size:20px;margin-top:14px">{s["n"]}</div>'
            f'<div class="box" style="position:relative;width:1171px;box-sizing:border-box;background:{c["bg"]};'
            f'border:2px solid {c["border"]};border-radius:8px;padding:22px 30px">{pill}{head}{body}</div></div>')


CONN = ('<div class="conn n" style="margin-left:66px;width:1171px;display:flex;flex-direction:column;align-items:center;'
        'height:60px"><div class="shaft"><span class="clip"><i></i></span></div><div style="border-left:5px solid '
        'transparent;border-right:5px solid transparent;border-top:9px solid #5a6270"></div></div>')


def rule(r):
    f, title, text, st, _ = r
    return (f'<div class="rule n" id="fr{f[1]}" data-f="{f}" style="display:flex;gap:16px;align-items:flex-start">'
            f'<div style="flex:none;width:44px;height:44px;box-sizing:border-box;border:2px solid #2a2f38;border-radius:50%;'
            f'display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:600;margin-top:2px">{f}</div>'
            f'<div><div style="font-size:23px;font-weight:600;margin-bottom:4px">{title} {status_chip(st)}</div>'
            f'<div style="font-size:19px;line-height:1.5;color:#2a2f38">{text}</div></div></div>')


def legend_item(swatch, label):
    return (f'<div style="display:flex;gap:10px;align-items:center"><div style="width:18px;height:18px;box-sizing:border-box;'
            f'{swatch};border-radius:3px"></div><span>{label}</span></div>')


workflow_html = (
    '<div class="stage" id="wf-stage" data-w="1920" style="width:1920px;box-sizing:border-box;padding:0 60px 0 0;'
    "font-family:'IBM Plex Sans','Helvetica Neue',Arial,sans-serif;color:#1f2430\">"
    '<div style="display:flex;gap:80px;align-items:stretch"><div style="display:flex;flex-direction:column">'
    + CONN.join(wf_step(s) for s in WF_STEPS)
    + '</div><div class="rules" style="width:454px;box-sizing:border-box;border:2px solid #2a2f38;border-radius:12px;'
    'padding:30px 28px;display:flex;flex-direction:column"><div style="font-size:27px;font-weight:600;margin-bottom:30px">'
    'Fairness rules</div><div style="flex:1;display:flex;flex-direction:column;justify-content:space-between;'
    'padding-bottom:60px">' + "".join(rule(r) for r in RULES) + "</div></div></div>"
    '<div style="display:flex;gap:56px;margin:56px 0 0 66px;font-size:18px;color:#2a2f38">'
    + legend_item("background:#e9f0fa;border:2px solid #4a74b5", "Data")
    + legend_item("background:#ececec;border:2px solid #888", "Frozen")
    + legend_item("background:#fbefd9;border:2px solid #c08a2a", "Trained")
    + legend_item("background:#e6f0e8;border:2px solid #3d8a4f", "Evaluation")
    + legend_item("background:#fff;border:2px solid #2a2f38", "Fairness rules")
    + "</div></div>"
)

# ----------------------------------------------------------------------------------------------
# Section 2: model architecture (the design's absolute layout, one column per arm)
# ----------------------------------------------------------------------------------------------

COLW = 300
X = {"tm": 244, "th": 568, "pr": 892, "ts": 1240, "ae": 1564, "un": 1912}
CX = {k: v + COLW // 2 for k, v in X.items()}
RIGHT = X["un"] + COLW            # 2212
TG = (X["tm"], X["pr"] + COLW)    # token grid span
PX = (X["ts"], X["ae"] + COLW)    # per-pixel span
STAGE_W, STAGE_H = 2540, 1945
HATCH = "repeating-linear-gradient(135deg,#f4f4f4 0 7px,#e8e8e8 7px 9px)"

parts: list[str] = []


def add(s):
    parts.append(s)


def arrow(x, top, h, lanes, row, small=False, extra=""):
    """A vertical arrow in design pixels; small ones sit inside a backbone box."""
    w, tip = ("1.5px", "border-left:5px solid transparent;border-right:5px solid transparent;border-top:8px solid #222;left:-4px") \
        if small else ("2px", "border-left:6px solid transparent;border-right:6px solid transparent;border-top:11px solid #222;left:-5px")
    cls = "arrow" if small else "arrow n"
    delay = round(top / STAGE_H * 1.6, 2)
    return (f'<div class="{cls}" data-lanes="{lanes}" data-row="{row}" style="left:{x - 1}px;top:{top}px;width:{w};'
            f'height:{h}px;--travel:{h + 40}px;--d:{delay}s{extra}"><span class="clip"><i></i></span>'
            f'<b class="tip" style="{tip}"></b></div>')


def box(id_, x, y, w, h, lanes, row, style, content, extra_cls=""):
    return (f'<div class="n box {extra_cls}" id="{id_}" data-lanes="{lanes}" data-row="{row}" style="position:absolute;'
            f'left:{x}px;top:{y}px;width:{w}px;height:{h}px;box-sizing:border-box;{style}">{content}</div>')


DATA = "background:#e3f2fd;border:1.5px solid #3a6fb0;border-radius:3px"
TRAIN = "background:#fdefd6;border:1.5px solid #c0802a;border-radius:3px"
EVAL = "background:#e2f5e4;border:1.5px solid #3f9a52;border-radius:3px"

# Group headers and model names
for label, (a, b) in [("Frozen GFM, token grid", TG), ("Frozen GFM, per-pixel embeddings", PX),
                      ("Baseline", (X["un"], RIGHT))]:
    add(f'<div class="grp" style="left:{a}px;width:{b - a}px">{label}</div>'
        f'<div style="position:absolute;left:{a}px;top:44px;width:{b - a}px;height:8px;box-sizing:border-box;'
        f'border:1px solid #999;border-bottom:none"></div>')
NAMES = {"tm": "TerraMind", "th": "THOR", "pr": "Prithvi-EO-2.0", "ts": "TESSERA", "ae": "AlphaEarth", "un": "U-Net"}
for k, name in NAMES.items():
    add(f'<button class="lane-btn n" data-lane="{k}" data-lanes="{k}" style="left:{X[k]}px;width:{COLW}px">{name}</button>'
        f'<div style="position:absolute;left:{X[k]}px;top:108px;width:{COLW}px;height:2px;background:#222"></div>')

# Row labels
for label, top, w in [("DATA SOURCE", 168, 213), ("PREPROCESSING", 294, 236), ("MODEL INPUT", 421, 213),
                      ("BACKBONE", 779, 213), ("EMBEDDINGS<br>ON DISK", 1094, 213), ("NECK", 1323, 213),
                      ("DECODER", 1516, 213), ("HEAD", 1635, 213), ("OUTPUT", 1765, 213)]:
    add(f'<div class="rowlab" style="width:{w}px;top:{top}px">{label}</div>')


def three_lines(lines, size=20):
    tops = [168 - 139 - 2, 294 - 139 - 2, 421 - 139 - 2]
    return "".join(f'<div class="dl" style="top:{t}px;font-size:{size}px">{l}</div>' for t, l in zip(tops, lines))


# Data sources
add(box("src-comp", X["tm"], 139, X["ts"] + COLW - X["tm"], 335, "tm th pr ts", "src", DATA, three_lines([
    "Sentinel-2 L2A + Sentinel-1 RTC, 2021, from the Planetary Computer",
    "cloud and shadow screening · monthly median · gap filling · EPSG:3035, 10 m",
    "224 × 224 px chips (2.24 km) · 12 monthly composites · crop mask + parcel ids"])))
add(box("src-ae", X["ae"], 139, COLW, 335, "ae", "src", DATA, three_lines([
    "Raw extraction of precomputed embeddings", "AlphaEarth v1, annual 2021", "64-d per pixel, on the same chips"], 17)))
add(box("src-un", X["un"], 139, COLW, 335, "un", "src", DATA, three_lines([
    "the same S2 + S1 composites", "chip-set statistics", "12 months × 14 bands = 168 channels"], 17)))

# Arrows into the backbone row, and the two lanes that pass it
TMTH = X["tm"] + (2 * COLW + 24) // 2
add(arrow(TMTH, 474, 79, "tm th", "bb"))
add(arrow(CX["pr"], 474, 79, "pr", "bb"))
add(arrow(CX["ts"], 474, 79, "ts", "bb"))
add(arrow(CX["ae"], 474, 596, "ae", "disk"))
add(arrow(CX["un"], 474, 740, "un", "dec"))


def months_tall(left, top, width):
    cols = "".join(
        f'<div style="width:30px;display:flex;flex-direction:column;align-items:center"><div style="width:30px;height:96px;'
        f'box-sizing:border-box;border:1.5px solid #555;border-radius:2px;background:#fff;display:flex;align-items:center;'
        f'justify-content:center"><span style="transform:rotate(-90deg);white-space:nowrap;font-size:15px">month {m}</span>'
        f'</div><div style="width:1.5px;height:13px;background:#555"></div><div style="width:14px;height:14px;'
        f'background:#555"></div><div style="width:1.5px;height:14px;background:#555"></div></div>' for m in range(1, 13))
    return (f'<div style="position:absolute;left:{left}px;top:{top}px;width:{width}px;display:flex;'
            f'justify-content:space-between">{cols}</div>')


def months_small(left, top, width, tick=20):
    cols = "".join(
        f'<div style="width:20px;display:flex;flex-direction:column;align-items:center"><div style="width:20px;height:24px;'
        f'box-sizing:border-box;border:1.5px solid #555;border-radius:2px;background:#fff;display:flex;align-items:center;'
        f'justify-content:center;font-size:11px">{m}</div><div style="width:1.5px;height:{tick}px;background:#555"></div>'
        f'</div>' for m in range(1, 13))
    return (f'<div style="position:absolute;left:{left}px;top:{top}px;width:{width}px;display:flex;'
            f'justify-content:space-between">{cols}</div>')


def pill(left, top, width, text, size):
    return (f'<div style="position:absolute;left:{left}px;top:{top}px;width:{width}px;height:32px;box-sizing:border-box;'
            f'background:#fff;border:1.5px solid #888;border-radius:16px;display:flex;align-items:center;'
            f'justify-content:center;font-size:{size}px">{text}</div>')


FROZEN = f"border:1.5px solid #888;border-radius:2px;background:{HATCH}"

# TerraMind and THOR: the TemporalWrapper of the design
W2 = 2 * COLW + 24
L = (W2 - 436) // 2
add(box("bb-tmth", X["tm"], 553, W2, 475, "tm th", "bb", FROZEN,
        pill((W2 - 238) // 2, 16, 238, "S2 bands + S1 VV, VH", 17)
        + arrow(W2 // 2 + 1, 48, 22, "tm th", "bb", small=True)
        + f'<div style="position:absolute;left:{L}px;top:70px;width:436px;height:230px;box-sizing:border-box;background:#fff;'
          f'border:1.5px solid #555;border-radius:2px"></div>'
        + f'<div style="position:absolute;left:{L}px;top:79px;width:436px;text-align:center;font-weight:700">TemporalWrapper</div>'
        + months_tall(L + 13, 116, 410)
        + f'<div style="position:absolute;left:{L + 13}px;top:253px;width:410px;height:34px;box-sizing:border-box;'
          f'background:#fff;border:1.5px solid #555;border-radius:2px;display:flex;align-items:center;justify-content:center;'
          f'font-size:17px"><span><b>concatenation</b> → 12 × D channels</span></div>'
        + f'<div style="position:absolute;left:{L}px;top:313px;width:436px;height:150px;box-sizing:border-box;background:#fff;'
          f'padding:10px 14px;font-size:17px;line-height:1.42">Single-date model: (1) months folded into batch, (2) one '
          f'frozen ViT pass per month, S2 and S1 tokens averaged, no cross-month attention, (3) concat → 12 × D on 14 × 14 '
          f'grid (~160 m). Same handling for both models.</div>'))

# Prithvi: the design's single ViT encoder
add(box("bb-pr", X["pr"], 553, COLW, 475, "pr", "bb", FROZEN,
        pill(30, 16, 240, "S2 (6 bands) + date, location", 15)
        + arrow(151, 48, 22, "pr", "bb", small=True)
        + '<div style="position:absolute;left:0;top:72px;width:300px;text-align:center;font-size:14px;color:#555">12 months</div>'
        + months_small(15, 97, 270)
        + '<div style="position:absolute;left:12px;top:141px;width:276px;height:129px;box-sizing:border-box;background:#fff;'
          'border:1.5px solid #555;border-radius:2px;display:flex;align-items:center;justify-content:center;font-weight:700">'
          'single ViT encoder</div>'
        + '<div style="position:absolute;left:12px;top:287px;width:276px;height:176px;box-sizing:border-box;background:#fff;'
          'padding:9px 12px;font-size:14px;line-height:1.4">Spatio-temporal model: (1) 3D patch embedding of all 12 months, '
          '(2) one frozen ViT pass over 12 × 256 tokens, joint space-time attention, (3) fused output reshaped → 12 × D on '
          '16 × 16 grid.</div>'))

# TESSERA: the same drawing language, one pixel at a time
small_box = ('position:absolute;box-sizing:border-box;background:#fff;border:1.5px solid #555;border-radius:2px;'
             'display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center')
add(box("bb-ts", X["ts"], 553, COLW, 475, "ts", "bb", FROZEN,
        pill(30, 16, 240, "S2 + S1, with day of year", 15)
        + arrow(151, 48, 22, "ts", "bb", small=True)
        + '<div style="position:absolute;left:0;top:72px;width:300px;text-align:center;font-size:14px;color:#555">'
          '12 months of one 10 m pixel</div>'
        + months_small(15, 97, 270, tick=10)
        + '<div style="position:absolute;left:25px;top:131px;width:250px;height:1.5px;background:#555"></div>'
        + arrow(79, 132, 18, "ts", "bb", small=True) + arrow(222, 132, 18, "ts", "bb", small=True)
        + f'<div style="{small_box};left:12px;top:150px;width:132px;height:64px"><b style="font-size:15px">S2 Transformer'
          f'</b><span style="font-size:13px;color:#444">10 bands</span></div>'
        + f'<div style="{small_box};left:156px;top:150px;width:132px;height:64px"><b style="font-size:15px">S1 Transformer'
          f'</b><span style="font-size:13px;color:#444">VV, VH</span></div>'
        + arrow(79, 214, 14, "ts", "bb", small=True) + arrow(222, 214, 14, "ts", "bb", small=True)
        + f'<div style="{small_box};left:12px;top:228px;width:276px;height:30px;font-size:14px">attention pooling over time</div>'
        + arrow(151, 258, 14, "ts", "bb", small=True)
        + f'<div style="{small_box};left:12px;top:272px;width:276px;height:30px;font-size:14px"><span><b>concat + MLP</b> → '
          f'128-d</span></div>'
        + '<div style="position:absolute;left:12px;top:318px;width:276px;height:145px;box-sizing:border-box;background:#fff;'
          'padding:9px 12px;font-size:14px;line-height:1.4">Pixel-wise temporal model: (1) each pixel\'s 12-month S2 and S1 '
          'series, (2) one Transformer per sensor over time, attention-pooled, (3) fused → 128-d per 10 m pixel. No spatial '
          'context.</div>'))

# Out of the backbone row
add(arrow(TMTH, 1028, 42, "tm th", "disk"))
add(arrow(CX["pr"], 1028, 42, "pr", "disk"))
add(arrow(CX["ts"], 1028, 42, "ts", "disk"))

# Embeddings on disk: the design's double-line bar, over the five GFM lanes
DISK_W = PX[1] - X["tm"]
add(box("disk", X["tm"], 1070, DISK_W, 100, "tm th pr ts ae", "disk",
        "background:#f7f7f5;border:5px double #444;border-radius:3px",
        f'<div style="position:absolute;left:-5px;top:-5px;width:{TG[1] - TG[0]}px;height:100px;display:flex;'
        f'align-items:center;justify-content:center;text-align:center;padding:0 40px;box-sizing:border-box;line-height:1.45">'
        f'<span><b>Encode once:</b> features saved per chip (training chips also in 8 flipped and rotated copies)</span></div>'
        f'<div style="position:absolute;left:{PX[0] - 24 - X["tm"] - 5}px;top:13px;height:64px;border-left:1.5px dashed #999">'
        f'</div><div style="position:absolute;left:{PX[0] - X["tm"] - 5}px;top:-5px;width:{PX[1] - PX[0]}px;height:100px;'
        f'display:flex;align-items:center;justify-content:center;text-align:center;padding:0 30px;box-sizing:border-box;'
        f'line-height:1.45"><span><b>Encode once</b> (TESSERA) or <b>download once</b> (AlphaEarth): embeddings saved per '
        f'chip</span></div>'))

# Token-grid lanes go through the neck; TESSERA and AlphaEarth split off straight to the pixel-wise MLP
for k in ("tm", "th", "pr"):
    add(arrow(CX[k], 1170, 44, k, "neck"))
for k in ("ts", "ae"):
    add(arrow(CX[k], 1170, 330, k, "dec"))

NECK_STACK = "".join(
    (f'<div style="width:464px;height:38px;box-sizing:border-box;background:#fff;border:1.5px solid #c0802a;display:flex;'
     f'align-items:center;justify-content:center">{t}</div>'
     + ('' if i == 3 else '<div style="width:1.5px;height:20px;background:#222;position:relative"><div style="position:'
        'absolute;bottom:0;left:-4px;border-left:5px solid transparent;border-right:5px solid transparent;border-top:8px '
        'solid #222"></div></div>'))
    for i, t in enumerate(["SelectIndices (4 layers)", "ReshapeTokensToImage", "ChannelBottleneck → 768",
                           "LearnedInterpolateToPyramidal (×4, ×2, ×1, ×½)"]))
add(box("neck", X["tm"], 1214, TG[1] - TG[0], 242, "tm th pr", "neck", TRAIN,
        f'<div style="position:absolute;left:{(TG[1] - TG[0] - 464) // 2}px;top:16px;display:flex;flex-direction:column;'
        f'align-items:center">{NECK_STACK}</div>'))

# The U-Net baseline, trained end to end, beside the neck and decoder rows
UN_STACK = ('<div style="position:absolute;left:0;top:18px;width:100%;text-align:center;font-weight:700">ResNet-18 U-Net</div>'
            '<div style="position:absolute;left:18px;top:58px;width:264px;display:flex;flex-direction:column;align-items:center">'
            '<div style="width:264px;height:38px;box-sizing:border-box;background:#fff;border:1.5px solid #c0802a;display:flex;'
            'align-items:center;justify-content:center">ResNet-18 encoder, 11.7 M</div>'
            '<div style="width:1.5px;height:20px;background:#222;position:relative"><div style="position:absolute;bottom:0;'
            'left:-4px;border-left:5px solid transparent;border-right:5px solid transparent;border-top:8px solid #222"></div>'
            '</div><div style="width:264px;height:38px;box-sizing:border-box;background:#fff;border:1.5px solid #c0802a;'
            'display:flex;align-items:center;justify-content:center">U-Net decoder, 3.2 M</div></div>'
            '<div style="position:absolute;left:18px;top:200px;width:264px;font-size:16px;line-height:1.45;text-align:center;'
            'color:#333">Trained from scratch, end to end, with skip connections from every encoder stage.</div>')
add(box("unet", X["un"], 1214, COLW, 342, "un", "dec", TRAIN, UN_STACK))

for k in ("tm", "th", "pr"):
    add(arrow(CX[k], 1456, 44, k, "dec"))
add(box("dec-tg", X["tm"], 1500, TG[1] - TG[0], 56, "tm th pr", "dec",
        TRAIN + ";display:flex;align-items:center;justify-content:center",
        "<span><b>UNetDecoder</b> → 64 × 56 × 56 (Prithvi 64 × 64 × 64)</span>"))
add(box("dec-px", PX[0], 1500, PX[1] - PX[0], 56, "ts ae", "dec",
        TRAIN + ";display:flex;align-items:center;justify-content:center",
        "<span><b>Pixel-wise MLP</b> 128 or 64 → 512 → 256</span>"))

for k in X:
    add(arrow(CX[k], 1556, 44, k, "head"))
add(box("head", X["tm"], 1600, RIGHT - X["tm"], 94, "tm th pr ts ae un", "head", TRAIN,
        f'<div style="position:absolute;left:0;top:15px;width:100%;text-align:center"><b>SegmentationHead</b> → 20 class '
        f'logits</div><div style="position:absolute;left:0;top:54px;width:{TG[1] - TG[0]}px;text-align:center;color:#444">'
        f'bilinear upsampling → 224 × 224</div><div style="position:absolute;left:{TG[1] + 23 - X["tm"]}px;top:53px;'
        f'height:26px;border-left:1.5px dashed #c0802a"></div><div style="position:absolute;left:{PX[0] - X["tm"]}px;'
        f'top:54px;width:{RIGHT - PX[0]}px;text-align:center;color:#444">native 224 × 224</div>'))

for k in X:
    add(arrow(CX[k], 1694, 44, k, "out"))
add(box("out", X["tm"], 1738, RIGHT - X["tm"], 76, "tm th pr ts ae un", "out",
        EVAL + ";display:flex;align-items:center;justify-content:center;text-align:center",
        "<span><b>Crop-type segmentation</b>, evaluated identically for every lane: Macro-F1 per pixel and per parcel, "
        "on the same 1,475 test chips</span>"))

# The design's brackets
BX = RIGHT + 30
add(f'<div class="brk" id="brk-frozen" style="left:{BX}px;top:553px;height:617px"></div>'
    f'<div class="brk-t" style="left:{BX + 30}px;top:836px"><div style="font-weight:700">Runs once per model</div>'
    f'<div style="font-size:16px;color:#555">(frozen)</div></div>'
    f'<div class="brk" id="brk-trained" style="left:{BX}px;top:1214px;height:480px"></div>'
    f'<div class="brk-t" style="left:{BX + 30}px;top:1420px"><div style="font-weight:700">Repeated for every label budget, '
    f'draw and seed</div><div style="font-size:16px;color:#555">(trained)</div></div>')

# Legend
add(f'<div style="position:absolute;left:244px;top:1858px;width:{RIGHT - 244}px;height:1px;background:#ccc"></div>'
    '<div style="position:absolute;left:244px;top:1880px;display:flex;gap:40px;align-items:center">'
    + "".join(f'<div style="display:flex;gap:12px;align-items:center"><div style="width:34px;height:22px;box-sizing:border-box;'
              f'{sw};border-radius:2px"></div><span>{lab}</span></div>' for sw, lab in [
                  ("background:#e3f2fd;border:1.5px solid #3a6fb0", "data"),
                  ("background:repeating-linear-gradient(135deg,#f4f4f4 0 4px,#cfcfcf 4px 6px);border:1.5px solid #888", "frozen"),
                  ("background:#fdefd6;border:1.5px solid #c0802a", "trainable"),
                  ("background:#e2f5e4;border:1.5px solid #3f9a52", "evaluation"),
                  ("background:#f7f7f5;border:4px double #444", "double-line bar = saved to disk")])
    + "</div>")

architecture_html = (f'<div class="stage" id="ar-stage" data-w="{STAGE_W}" style="position:relative;width:{STAGE_W}px;'
                     f'height:{STAGE_H}px;font-family:\'Helvetica Neue\',Helvetica,Arial,sans-serif;color:#222;font-size:18px">'
                     + "".join(parts) + "</div>")

# ----------------------------------------------------------------------------------------------
# Section 3: the experiments of the analysis
# ----------------------------------------------------------------------------------------------

EXPERIMENTS = [
    dict(n=1, title="THOR token size: 160, 80 or 40 m", folder="2026-10-09_thor_token_size",
         setup="THOR v1 large on Sentinel-2, 733-chip sample, three seeds, 15 epochs.",
         result="40 m beats 160 m by about 0.05 Macro-F1 at every K and wins all nine paired fits; its cache is 16 times "
                "larger, about 25.5 TB for Estonia.",
         decision="160 m kept, for parity with TerraMind's 14 × 14 grid and a cache that fits a cluster node.",
         show=["bb-tmth"]),
    dict(n=2, title="Rotations: 1, 4 or 8 D4 copies", folder="2026-10-14_terramind_d4_count_ee_rev",
         setup="TerraMind v1 large on Sentinel-2, all of EE_2021_rev, three seeds, 15 epochs.",
         result="Eight copies beat the identity by 0.047, 0.061 and 0.070 at K = 100, 20 and 5 %; four sit between. Most of "
                "the gain comes through the checkpoint.",
         decision="Training chips are cached in all eight flipped and rotated copies.", show=["disk"]),
    dict(n=3, title="Temporal grid: 12 monthly or 24 half-month frames", folder="2026-10-12_terramind_12v24",
         setup="TerraMind v1 large on Sentinel-2, 733-chip sample, three seeds, 15 epochs.",
         result="+0.015, +0.035 and +0.001 for 24 frames, all within the seed spread, at twice the encoding cost; in late "
                "October 73 % of the chips have no image at all.",
         decision="Twelve monthly composites.", show=["src-comp", "bb-tmth"]),
    dict(n=4, title="TESSERA: its encoder on our composites, or the store's embeddings",
         folder="2026-10-14_tessera_composites",
         setup="Probe on EE_2021_rev_mini; fits on all of EE_2021_rev, three seeds.",
         result="The encoder on our composites matches the store in a probe (0.371 against 0.373). In the main workflow "
                "frozen v2 is 0.018, 0.037 and 0.068 below the store arm.",
         decision="TESSERA v2, frozen, on our composites: no 0.7 px offset, the same imagery as every arm.",
         show=["bb-ts", "src-comp"]),
    dict(n=5, title="TerraMind: Sentinel-2, Sentinel-1 or both", folder="2026-10-13_terramind_s1_s2",
         setup="TerraMind v1 large, all of EE_2021_rev, three seeds, 15 epochs.",
         result="Both sensors beat Sentinel-2 alone by 0.026, 0.054 and 0.025 and win every fit; Sentinel-1 alone is 0.02 "
                "to 0.07 below Sentinel-2.",
         decision="Every model reads every sensor it was pretrained on.", show=["src-comp", "bb-tmth"]),
    dict(n=6, title="Fine-tuned or frozen encoder", folder="2026-10-15_kshot_finetune",
         setup="Every encoder with weights, all of EE_2021_rev, 45 epochs, one seed.",
         result="TESSERA gains 0.04 to 0.08; TerraMind, THOR and Prithvi about 0.01 in validation, their test difference "
                "set by the checkpoint. Fine-tuning costs 4 to 15 times the frozen fit.",
         decision="Encoders stay frozen in the main workflow; fine-tuning is reported as an insight.",
         show=["bb-tmth", "bb-pr", "bb-ts", "brk-frozen"]),
]


def exp_card(e):
    url = f"{REPO}/experiments/{e['folder']}/REPORT.md"
    return (f'<article class="exp"><div class="exp-num">{e["n"]}</div><div class="exp-box">'
            f'<a class="rep" href="{url}" target="_blank" rel="noopener">REPORT.md ↗</a>'
            f'<h3><a href="{url}" target="_blank" rel="noopener">{E(e["title"])}</a></h3>'
            f'<p class="setup">{E(e["setup"])}</p>'
            f'<p><b>Result.</b> {E(e["result"])}</p><p><b>Decision.</b> {E(e["decision"])}</p>'
            f'<button class="show" type="button" data-exp="{e["n"]}">Show in the architecture</button></div></article>')


estonia_url = f"{REPO}/docs/phase1/results_estonia/REPORT.md"
experiments_html = (
    '<p class="lead">Each experiment varies one factor on one encoder and compares the variants fit by fit, on shared '
    'draws and seeds. The relative result backs the decision, so none is rerun on the current arms; experiments 1 to 3 ran '
    'on TerraMind or THOR with Sentinel-2 alone.</p><div class="exp-list">'
    + "".join(exp_card(e) for e in EXPERIMENTS)
    + f'<article class="exp res"><div class="exp-num">R</div><div class="exp-box"><a class="rep" href="{estonia_url}" '
      f'target="_blank" rel="noopener">REPORT.md ↗</a><h3><a href="{estonia_url}" target="_blank" rel="noopener">First '
      f'results for Estonia 2021</a></h3><p>The main workflow on all of Estonia: every arm frozen and fine-tuned, per pixel '
      f'and per parcel, beside the published Estonian results.</p></div></article></div>')

# ----------------------------------------------------------------------------------------------
# Interactions: step-through, lane focus, live flow, zoom
# ----------------------------------------------------------------------------------------------

LANES = {
    "tm": ("TerraMind v1 large", [
        ("Input", "S2 12 bands + S1 VV, VH (S1 RTC)"),
        ("Encoder", "frozen, one pass per month; S2 and S1 tokens averaged at every position"),
        ("Tokens", "14 × 14 per month (16 px patches, about 160 m)"),
        ("Layers kept", "5, 11, 17, 23"),
        ("Decoder", "ChannelBottleneck 768 + UNetDecoder"),
        ("Trained", "53.3 M parameters"),
        ("Optimiser", "learning rate 1e-4, batch 2 accumulated twice")]),
    "th": ("THOR v1 large", [
        ("Input", "S2 10 bands (B01, B09 left out) + S1 VV, VH"),
        ("Encoder", "frozen, one pass per month; three band groups (S2 10 m, S2 20 m, S1) averaged"),
        ("Tokens", "160 m: 16 px on the 10 m bands, 8 px on the 20 m bands, 14 × 14"),
        ("Layers kept", "5, 11, 17, 23"),
        ("Decoder", "ChannelBottleneck 768 + UNetDecoder, exactly TerraMind's"),
        ("Trained", "53.3 M parameters"),
        ("Optimiser", "learning rate 1e-4, batch 2 accumulated twice")]),
    "pr": ("Prithvi-EO-2.0 600M TL", [
        ("Input", "S2 6 HLS bands + date and location"),
        ("Encoder", "frozen, 12 months in one ViT, joint space-time attention"),
        ("Tokens", "16 × 16 per month (14 px patches)"),
        ("Layers kept", "7, 15, 23, 31"),
        ("Decoder", "ChannelBottleneck 768 + UNetDecoder"),
        ("Trained", "62.7 M parameters (12 × 1,280 channels into the bottleneck)"),
        ("Optimiser", "learning rate 1e-4, batch 2 accumulated twice")]),
    "ts": ("TESSERA v2 Medium", [
        ("Input", "S2 10 bands + S1 VV, VH, each month dated the 15th"),
        ("Encoder", "frozen, 21.0 M; one Transformer per sensor over time, per pixel"),
        ("Output", "128-d per 10 m pixel, computed once per chip set"),
        ("Decoder", "per-pixel MLP, 512 and 256"),
        ("Trained", "0.20 M parameters"),
        ("Optimiser", "learning rate 1e-3, batch 4")]),
    "ae": ("AlphaEarth v1", [
        ("Input", "Google's annual 64-d embedding of 2021"),
        ("Encoder", "none run here: precomputed and downloaded once"),
        ("Output", "64-d per 10 m pixel, warped onto the chip grid"),
        ("Decoder", "per-pixel MLP, 512 and 256"),
        ("Trained", "0.17 M parameters"),
        ("Optimiser", "learning rate 1e-3, batch 4")]),
    "un": ("U-Net from scratch, the baseline", [
        ("Input", "S2 + S1 on 168 channels (12 months × 14 bands)"),
        ("Model", "ResNet-18 U-Net, randomly initialised"),
        ("Trained", "all 14.8 M parameters: encoder 11.7 M, decoder 3.2 M"),
        ("Optimiser", "learning rate 1e-3, batch 4")]),
}
SHARED = ("Every arm: 45 epochs at every K · AdamW, weight decay 0.05 · effective batch 4 chips · cross-entropy on "
          "labelled pixels · D4 augmentation · checkpoint of lowest validation loss.")

WALKS = {
    "wf": [dict(title=s["title"], text=s.get("walk", s["body"]), now=s["now"],
                targets=[f"#ws{s['n']}"] + [f"#fr{t[1]}" for t in s["tags"]]) for s in WF_STEPS],
    "ar": [
        dict(title="Data source", targets=["[data-row=src]"],
             text="Every lane reads the same chips and parcel masks. Five read the 12 monthly S2 + S1 composites; "
                  "AlphaEarth reads Google's precomputed embedding."),
        dict(title="Backbone", targets=["[data-row=bb]"],
             text="Frozen encoders. TerraMind and THOR run once per month, Prithvi reads all 12 months at once, TESSERA "
                  "reads each pixel's 12-month series."),
        dict(title="Embeddings on disk", targets=["[data-row=disk]"],
             text="Each encoder runs once per chip set and its output is saved. Token-grid training chips are saved in "
                  "8 flipped and rotated copies."),
        dict(title="Neck", targets=["[data-row=neck]"],
             text="Trained. Four encoder layers become maps, the 12 months are projected to 768 channels, then a 4-level "
                  "pyramid."),
        dict(title="Decoder", targets=["[data-row=dec]"],
             text="The token-grid lanes share one UNetDecoder of 12.9 M parameters. TESSERA and AlphaEarth go straight to "
                  "a per-pixel MLP. The U-Net trains end to end."),
        dict(title="Head", targets=["[data-row=head]"],
             text="A segmentation head gives 20 class logits per 10 m pixel; token-grid outputs are upsampled to "
                  "224 × 224."),
        dict(title="Output", targets=["[data-row=out]"],
             text="Every lane is scored the same way on the 1,475 test chips: Macro-F1 per pixel, the primary metric, and "
                  "per parcel."),
    ],
}
SHOW = {str(e["n"]): dict(title=f"Experiment {e['n']}: {e['title']}", text=e["decision"],
                          targets=["#" + t for t in e["show"]]) for e in EXPERIMENTS}

CSS = r"""
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:#fff;color:#1f2430;font-family:'IBM Plex Sans','Helvetica Neue',Arial,sans-serif}
a{color:#2b5c9e}a:hover{color:#1a3d6e}
.top{display:flex;flex-wrap:wrap;gap:8px 24px;align-items:baseline;justify-content:space-between;padding:20px 24px 0;
  font-size:14px;color:#555}
.top b{color:#222;font-size:16px}
.top nav a{margin-left:16px;text-decoration:none}
section{padding:40px 16px 24px}
h2.sec{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-size:40px;font-weight:700;color:#222;text-align:center;
  margin:0 0 14px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:center;margin:0 0 22px;font-size:14px}
.toolbar button,.walk button,.card button{font:inherit;font-size:14px;font-weight:600;background:#fff;color:#2a2f38;
  border:2px solid #2a2f38;border-radius:16px;padding:4px 14px;cursor:pointer}
.toolbar button:hover,.walk button:hover{background:#f1f5fb}
.toolbar button[aria-pressed=true]{background:#2a2f38;color:#fff}
.toolbar .hint{color:#666;margin-left:6px}
.viewport{position:relative;width:100%;overflow-x:auto;overflow-y:hidden}
.stage{position:absolute;left:0;top:0;transform-origin:0 0}
/* focus */
.n{transition:opacity .25s,box-shadow .25s}
.stage.focusing .n{opacity:.16}
.stage.focusing.soft .n{opacity:.35}
.stage.focusing .n.hl{opacity:1}
.stage.focusing .box.hl{box-shadow:0 0 0 6px rgba(63,109,176,.35)}
.stage.focusing .wstep.hl .box{box-shadow:0 0 0 6px rgba(63,109,176,.35)}
.stage.focusing .rule.hl{background:#f1f5fb;border-radius:8px}
/* workflow */
.fpill{position:absolute;right:18px;top:18px;border:2px solid #2a2f38;border-radius:16px;padding:3px 16px;font-size:15px;
  font-weight:600;background:#fff}
.fpill span{cursor:default}
.shaft{position:relative;flex:1;width:2px;background:#5a6270}
.shaft-h{position:relative;flex:1;height:2px;background:#5a6270}
.rule{cursor:default;padding:4px}
.st{display:none;font-size:14px;font-weight:600;border-radius:12px;padding:2px 10px;vertical-align:middle;margin-left:8px}
.show-status .st{display:inline-block}
.wstep .st{margin-left:14px}
.st-met{background:#e6f0e8;color:#2e6b3a;border:1.5px solid #3d8a4f}
.st-part{background:#fbefd9;color:#8a5a14;border:1.5px solid #c08a2a}
.st-no{background:#fbe9e7;color:#9b2c1f;border:1.5px solid #c0574a}
/* architecture */
.grp{position:absolute;top:18px;text-align:center;font-size:15px;font-weight:700;letter-spacing:1.2px;color:#666;
  text-transform:uppercase}
.lane-btn{position:absolute;top:62px;height:44px;background:none;border:0;border-radius:6px;font:700 24px 'Helvetica Neue',
  Helvetica,Arial,sans-serif;color:#222;cursor:pointer}
.lane-btn:hover{background:#f1f5fb}
.lane-btn[aria-pressed=true]{background:#e3f2fd}
.rowlab{position:absolute;left:0;text-align:right;font-weight:700;letter-spacing:1.5px;color:#444;line-height:1.35}
.dl{position:absolute;left:0;width:100%;text-align:center;padding:0 24px;line-height:1.35}
.arrow{position:absolute;background:#222}
.arrow .tip{position:absolute;bottom:0}
.brk{position:absolute;width:14px;border:1.5px solid #222;border-left:none}
.brk-t{position:absolute;width:250px;line-height:1.35}
/* live flow */
.clip{position:absolute;display:none;overflow:hidden;pointer-events:none}
.live .clip{display:block}
.arrow .clip,.shaft .clip{left:-3px;width:8px;top:0;bottom:10px}
.shaft-h .clip{top:-3px;height:8px;left:0;right:0}
.clip i{position:absolute;left:0;top:-40px;width:8px;height:40px;border-radius:4px;
  background:linear-gradient(to bottom,rgba(63,109,176,0),rgba(63,109,176,.95) 60%,rgba(63,109,176,0));
  animation:flowY 2.4s linear infinite;animation-delay:var(--d,0s)}
.shaft .clip i{--travel:100px;animation-duration:1.6s}
.shaft-h .clip i{top:0;left:-40px;width:40px;height:8px;
  background:linear-gradient(to right,rgba(63,109,176,0),rgba(63,109,176,.95) 60%,rgba(63,109,176,0));animation-name:flowX}
@keyframes flowY{from{transform:translateY(0)}to{transform:translateY(var(--travel,120px))}}
@keyframes flowX{from{transform:translateX(0)}to{transform:translateX(110px)}}
@media (prefers-reduced-motion:reduce){.clip i{animation:none;display:none}}
/* experiments */
#experiments .lead{max-width:1100px;margin:0 auto 24px;font-size:17px;line-height:1.55;color:#2a2f38;text-align:center}
.exp-list{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
.exp{display:flex;gap:16px;align-items:flex-start}
.exp-num{flex:none;width:40px;height:40px;border-radius:50%;background:#8a5a14;color:#fff;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:18px;margin-top:12px}
.exp-box{position:relative;flex:1;min-width:0;background:#fcf6ec;border:2px solid #b9862a;border-radius:8px;
  padding:18px 24px 16px}
.exp.res .exp-num{background:#2e6b3a}.exp.res .exp-box{background:#e6f0e8;border-color:#3d8a4f}
.exp h3{margin:0 120px 8px 0;font-size:21px;font-weight:600}
.exp h3 a{color:#1f2430;text-decoration:none}.exp h3 a:hover{text-decoration:underline}
.exp p{margin:6px 0;font-size:16px;line-height:1.5;color:#2a2f38}
.exp p.setup{color:#666;font-size:15px}
.rep{position:absolute;right:16px;top:16px;border:2px solid #2a2f38;border-radius:16px;padding:3px 14px;font-size:14px;
  font-weight:600;background:#fff;color:#2a2f38;text-decoration:none}
.rep:hover{background:#2a2f38;color:#fff}
.show{margin-top:8px;font:inherit;font-size:14px;font-weight:600;background:none;border:0;padding:0;color:#2b5c9e;
  cursor:pointer;text-decoration:underline}
/* walk bar and lane card */
.walk{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);width:min(760px,calc(100% - 24px));background:#fff;
  border:2px solid #2a2f38;border-radius:12px;padding:16px 20px;box-shadow:0 8px 28px rgba(0,0,0,.18);z-index:60}
.walk[hidden],.card[hidden]{display:none}
.walk .count{font-size:13px;color:#666;font-weight:600;letter-spacing:.5px}
.walk h4{margin:2px 0 6px;font-size:20px}
.walk p{margin:4px 0;font-size:15px;line-height:1.5}
.walk p.now{color:#2e6b3a}
.walk .nav{display:flex;gap:8px;justify-content:flex-end;margin-top:10px}
.card{position:fixed;right:18px;top:18px;width:min(380px,calc(100% - 24px));background:#fff;border:2px solid #2a2f38;
  border-radius:12px;padding:16px 20px;box-shadow:0 8px 28px rgba(0,0,0,.18);z-index:61;font-size:14px}
.card h4{margin:0 0 10px;font-size:19px;padding-right:70px}
.card dl{display:grid;grid-template-columns:96px 1fr;gap:6px 10px;margin:0}
.card dt{color:#666;font-weight:600}.card dd{margin:0;line-height:1.45}
.card .shared{margin:12px 0 0;color:#555;line-height:1.45;border-top:1px solid #ddd;padding-top:10px}
.card button{position:absolute;right:14px;top:14px;padding:2px 12px}
@media (max-width:700px){h2.sec{font-size:28px}.exp h3{margin-right:0;padding-top:30px}.card{top:auto;bottom:18px}}
"""

JS = r"""
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const DATA = JSON.parse($('#page-data').textContent);

function fit(vp) {
  const stage = $('.stage', vp), w = +stage.dataset.w;
  const s = vp.dataset.zoom === 'full' ? 1 : Math.min(1, vp.clientWidth / w);
  stage.style.transform = `scale(${s})`;
  stage.style.left = Math.max(0, (vp.clientWidth - w * s) / 2) + 'px';
  vp.style.height = Math.ceil(stage.offsetHeight * s) + 'px';
}
const fitAll = () => $$('.viewport').forEach(fit);
addEventListener('resize', fitAll);
addEventListener('load', fitAll);
document.fonts && document.fonts.ready.then(fitAll);
fitAll();

function focus(stage, els, soft) {
  $$('.hl', stage).forEach(e => e.classList.remove('hl'));
  stage.classList.add('focusing');
  stage.classList.toggle('soft', !!soft);
  els.forEach(e => e.classList.add('hl'));
}
function clear(stage) {
  stage.classList.remove('focusing', 'soft');
  $$('.hl', stage).forEach(e => e.classList.remove('hl'));
}
function reveal(el) {
  const r = el.getBoundingClientRect();
  if (r.top < 60 || r.bottom > innerHeight - 200) scrollBy({top: r.top - innerHeight * 0.3, behavior: 'smooth'});
}

// live flow
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
function setLive(on) {
  document.body.classList.toggle('live', on);
  $$('[data-act=live]').forEach(b => { b.textContent = on ? 'Live' : 'Still'; b.setAttribute('aria-pressed', on); });
}
setLive(!reduce);

// walk-through
const walk = $('#walk');
let W = null;
function showStep() {
  const step = W.steps[W.i], stage = $('#' + W.stage + '-stage');
  $('.count', walk).textContent = W.steps.length > 1 ? `Step ${W.i + 1} of ${W.steps.length}` : 'Highlighted';
  $('h4', walk).textContent = step.title;
  $('.text', walk).innerHTML = step.text || '';
  $('.text', walk).hidden = !step.text;
  $('.now', walk).textContent = step.now ? 'Now: ' + step.now : '';
  $('.now', walk).hidden = !step.now;
  $('[data-walk=prev]', walk).hidden = W.steps.length < 2;
  $('[data-walk=next]', walk).hidden = W.steps.length < 2;
  $('[data-walk=prev]', walk).disabled = W.i === 0;
  $('[data-walk=next]', walk).textContent = W.i === W.steps.length - 1 ? 'Done' : 'Next';
  const els = step.targets.flatMap(sel => $$(sel, stage));
  focus(stage, els);
  if (els[0]) reveal(els[0]);
}
function startWalk(stage, steps, i = 0) {
  closeCard();
  if (W) clear($('#' + W.stage + '-stage'));
  W = {stage, steps, i};
  walk.hidden = false;
  showStep();
}
function endWalk() {
  if (!W) return;
  clear($('#' + W.stage + '-stage'));
  W = null;
  walk.hidden = true;
}
walk.addEventListener('click', e => {
  const a = e.target.dataset.walk;
  if (!a || !W) return;
  if (a === 'close') return endWalk();
  if (a === 'prev' && W.i > 0) { W.i--; showStep(); }
  if (a === 'next') { if (W.i < W.steps.length - 1) { W.i++; showStep(); } else endWalk(); }
});
addEventListener('keydown', e => {
  if (e.key === 'Escape') { endWalk(); closeCard(); }
  if (!W) return;
  if (e.key === 'ArrowRight' && W.i < W.steps.length - 1) { W.i++; showStep(); }
  if (e.key === 'ArrowLeft' && W.i > 0) { W.i--; showStep(); }
});

// toolbars
$$('.toolbar').forEach(tb => tb.addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  const act = b.dataset.act, vp = $('#' + tb.dataset.for + '-vp');
  if (act === 'live') setLive(!document.body.classList.contains('live'));
  if (act === 'walk') startWalk(tb.dataset.for, DATA.walks[tb.dataset.for]);
  if (act === 'zoom') {
    vp.dataset.zoom = vp.dataset.zoom === 'full' ? 'fit' : 'full';
    b.textContent = vp.dataset.zoom === 'full' ? '100 %' : 'Fit';
    b.setAttribute('aria-pressed', vp.dataset.zoom === 'full');
    fit(vp);
  }
  if (act === 'status') {
    const on = $('#wf-stage').classList.toggle('show-status');
    b.setAttribute('aria-pressed', on);
  }
}));

// workflow: fairness rules and the steps that carry them
const wf = $('#wf-stage');
function ruleFocus(f) {
  if (W) return;
  focus(wf, [...$$(`.rule[data-f=${f}]`, wf), ...$$('.wstep', wf).filter(s => s.dataset.tags.split(' ').includes(f))], true);
}
$$('.rule', wf).forEach(r => {
  r.addEventListener('mouseenter', () => ruleFocus(r.dataset.f));
  r.addEventListener('mouseleave', () => { if (!W) clear(wf); });
});
$$('.fpill span', wf).forEach(p => {
  p.addEventListener('mouseenter', () => ruleFocus(p.dataset.f));
  p.addEventListener('mouseleave', () => { if (!W) clear(wf); });
});

// architecture: follow one lane
const ar = $('#ar-stage'), card = $('#card');
let lane = null;
function closeCard() {
  if (!lane) return;
  lane = null;
  card.hidden = true;
  $$('.lane-btn', ar).forEach(b => b.setAttribute('aria-pressed', false));
  if (!W) clear(ar);
}
$$('.lane-btn', ar).forEach(b => b.addEventListener('click', () => {
  const k = b.dataset.lane;
  if (lane === k) return closeCard();
  endWalk();
  lane = k;
  $$('.lane-btn', ar).forEach(x => x.setAttribute('aria-pressed', x === b));
  focus(ar, $$('.n', ar).filter(e => (e.dataset.lanes || '').split(' ').includes(k)));
  const [title, rows] = DATA.lanes[k];
  $('h4', card).textContent = title;
  $('dl', card).innerHTML = rows.map(([t, d]) => `<dt>${t}</dt><dd>${d}</dd>`).join('');
  card.hidden = false;
}));
$('button', card).addEventListener('click', closeCard);

// experiments: show where they act in the architecture
$$('.show').forEach(b => b.addEventListener('click', () => {
  const s = DATA.show[b.dataset.exp];
  startWalk('ar', [s]);
}));
"""

DATA_JSON = json.dumps(dict(
    walks=WALKS,
    lanes={k: [t, [[a, b] for a, b in rows]] for k, (t, rows) in LANES.items()},
    show=SHOW,
), ensure_ascii=False).replace("</", "<\\/")


def toolbar(target, extra=""):
    return (f'<div class="toolbar" data-for="{target}"><button type="button" data-act="walk">▶ Step through</button>'
            f'<button type="button" data-act="live" aria-pressed="true">Live</button>'
            f'<button type="button" data-act="zoom" aria-pressed="false">Fit</button>{extra}</div>')


page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 1 pipeline</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<header class="top"><div><b>ExplainedGFM4Agri · Phase 1 pipeline</b> &nbsp;Main workflow <code>kshot</code>, Estonia 2021,
22 October 2026</div><nav><a href="#workflow">1 · Workflow</a><a href="#architecture">2 · Architecture</a>
<a href="#experiments">3 · Experiments</a><a href="{REPO}/docs/phase1/pipeline.md" target="_blank" rel="noopener">pipeline.md ↗</a>
</nav></header>

<section id="workflow">
<h2 class="sec">1. Experimental workflow</h2>
{toolbar("wf", '<button type="button" data-act="status" aria-pressed="false">Status</button><span class="hint">Hover a fairness rule to see the steps that carry it.</span>')}
<div class="viewport" id="wf-vp">{workflow_html}</div>
</section>

<section id="architecture">
<h2 class="sec">2. Model architecture</h2>
{toolbar("ar", '<span class="hint">Click a model name to follow its lane.</span>')}
<div class="viewport" id="ar-vp">{architecture_html}</div>
</section>

<section id="experiments">
<h2 class="sec">3. Experiments that justify the pipeline</h2>
{experiments_html}
</section>

<div class="walk" id="walk" hidden role="dialog" aria-live="polite"><div class="count"></div><h4></h4>
<p class="text"></p><p class="now"></p><div class="nav"><button type="button" data-walk="prev">Back</button>
<button type="button" data-walk="next">Next</button><button type="button" data-walk="close">Close</button></div></div>
<aside class="card" id="card" hidden><button type="button">Close</button><h4></h4><dl></dl><p class="shared">{SHARED}</p></aside>

<script type="application/json" id="page-data">{DATA_JSON}</script>
<script>{JS}</script>
</body>
</html>
"""

OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT} ({len(page):,} bytes)")
