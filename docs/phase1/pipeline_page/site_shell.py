"""The site around the Phase 1 page: the landing at the site root, the Phase 2 and Phase 3 pages, and the parts every
page shares (the top bar with the three phases, the phase banner, the footer, the fonts and colours).

`build_page.py` imports this module (`site_shell`), builds the Phase 1 page with its pieces, and calls `write_site()` for the rest.
Everything the pages say comes from `docs/thesis_design.md` and the documents the Phase 1 page already shows.
"""
from __future__ import annotations

import html
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/davidrers/ExplainedGMF4Agri"
REPO = f"{REPO_URL}/blob/main"
E = html.escape

TITLE = "Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions"
AUTHOR = "David Reyes"
PROGRAMME = "MSc thesis, M-GEO 2026–2027, GEO-AI track · ITC, University of Twente"
SUPERVISORS = "Dr. Mahdi Farnaghi (first) and Dr. Mariana Belgiu (second)"
TITLE_HTML = E(TITLE).replace("Label-", "Label\u2011")   # no line break inside "Label-Scarce"
TARGET = ("A user supplies an area of interest. The system returns a pixel-level crop segmentation of that area, "
          "pixel-level explanations of why each region was classified as it was, calibrated per-pixel confidence, and an "
          "LLM-written report traceable to that evidence.")

# Where each page sits, from the docs/ root that GitHub Pages serves
PAGES = {0: "index.html", 1: "phase1/pipeline_page/index.html", 2: "phase2/index.html", 3: "phase3/index.html"}
DEPTH = {0: "", 1: "../../", 2: "../", 3: "../"}

PHASES = {
    1: dict(name="GFMs Benchmark", topic="Label-efficient crop segmentation", so="SO1", status="active",
            status_text="In progress", tone="forest",
            rqs=[("RQ1", "How few labelled parcel polygons per class are required for GFM-based crop segmentation to "
                         "match a raw-feature baseline?")],
            now="Estonia 2021 runs end to end: six arms at K = 100, 20 and 5 %."),
    2: dict(name="XAI + UQ", topic="Explainability and uncertainty", so="SO2", status="planned",
            status_text="Not started", tone="forest",
            rqs=[("RQ2.1", "Can explainability methods identify the spectral and temporal information used by "
                           "GFM-based crop segmenters?"),
                 ("RQ2.2", "Can uncertainty quantification methods deliver reliable, calibrated confidence estimates?"),
                 ("RQ2.3", "Do the explanations and confidence estimates agree with the raw-feature baseline and with "
                           "agronomic knowledge?")],
            now="Planned; respecified once Phase 1 lands."),
    3: dict(name="LLM Report", topic="XAI-grounded LLM reporting", so="SO3", status="planned",
            status_text="Not started", tone="slate",
            rqs=[("RQ3", "Can XAI-derived information serve as effective structured context for an LLM generating "
                         "agricultural monitoring reports?")],
            now="Planned; the first part reduced if the schedule slips."),
}

# The planned set-ups of Phases 2 and 3, as docs/thesis_design.md states them
PLANNED = {
    2: [
        dict(title="The segmenters of Phase 1", kind="data",
             body="The GFM-based crop segmenters: pixel-level segmentation makes attribution natively per pixel, per "
                  "band and per date, a stronger basis for RQ2.1 than attributing a pooled vector."),
        [dict(title="Explainability", kind="method", rq="RQ2.1",
             body="Integrated Gradients, Occlusion and AttnLRP. The embedding tier applies to every model; the backbone "
                  "tier to the models with open weights: TerraMind, THOR and TESSERA v2."),
         dict(title="Uncertainty", kind="method", rq="RQ2.2",
             body="Monte Carlo Dropout as primary and Deep Ensembles as the heavier comparison, both at decoder level. "
                  "Reported as per-pixel predictive entropy, with reliability diagrams and Expected Calibration Error.")],
        dict(title="Agreement", kind="eval", rq="RQ2.3",
             body="The GFM importances, after binding to agronomic concepts, placed on the same axes as the baseline "
                  "importances."),
    ],
    3: [
        dict(title="From pixels to zones", kind="method",
             body="Pixel-level attribution cannot be passed to an LLM directly; aggregating it to zone summaries is a "
                  "Phase 3 design problem that Phase 1 must not foreclose."),
        dict(title="Context document", kind="data",
             body="A structured JSON document per target zone with three blocks: predictions, explanations, and external "
                  "context (ERA5 weather variables; country, elevation and SoilGrids soil class; image-texture "
                  "descriptors where available)."),
        dict(title="Report generation", kind="method",
             body="An LLM that follows the required output format reliably and is reproducible, generating at "
                  "temperature zero and constrained to the context document."),
        dict(title="Traceability", kind="eval", rq="RQ3",
             body="Each report scored for traceability against its source. If explanations fail their faithfulness "
                  "checks, the context document can still operate from predictions and calibrated uncertainty alone."),
    ],
}

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family='
         'IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">')

# ----------------------------------------------------------------------------------------------------------------------
# Icons, drawn: one stroke weight, forest ink with gold accents
# ----------------------------------------------------------------------------------------------------------------------

_S = 'fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"'
ICONS = {
    "satellite": ('<circle cx="23" cy="42" r="16" fill="currentColor"/><g fill="none" stroke="var(--paper)" '
                  'stroke-width="2.2"><ellipse cx="23" cy="42" rx="16" ry="6.5"/><ellipse cx="23" cy="42" rx="6.5" '
                  'ry="16"/></g><g transform="rotate(35 45 19)"><rect x="27" y="15" width="11" height="8" rx="1" '
                  'fill="currentColor"/><rect x="52" y="15" width="11" height="8" rx="1" fill="currentColor"/>'
                  '<path d="M32.5 15v8M57.5 15v8" stroke="var(--paper)" stroke-width="1.6"/><path d="M38 19h3M49 19h3" '
                  'stroke="currentColor" stroke-width="3"/><rect x="41" y="14" width="8" height="10" rx="1.5" '
                  'fill="var(--gold)"/></g><g fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round">'
                  '<path d="M41.9 27.5A9 9 0 0 1 36.5 22.1"/><path d="M40.2 32.2A14 14 0 0 1 31.8 23.8"/></g>'),
    "gfm": ('<path d="M26 20v-6M32 20v-6M38 20v-6M26 44v6M32 44v6M38 44v6M20 26h-6M20 32h-6M20 38h-6M44 26h6M44 32h6M44 38h6" '
            'stroke="currentColor" stroke-width="3"/><rect x="20" y="20" width="24" height="24" rx="3" fill="currentColor"/>'
            '<rect x="26" y="26" width="12" height="12" rx="1.5" fill="var(--gold)"/>'
            + "".join(f'<circle cx="{x}" cy="{y}" r="2.9" fill="{"var(--gold)" if 32 in (x, y) else "currentColor"}"/>'
                      for x, y in [(26, 11), (32, 11), (38, 11), (26, 53), (32, 53), (38, 53), (11, 26), (11, 32),
                                   (11, 38), (53, 26), (53, 32), (53, 38)])),
    "xai": (f'<g {_S}><circle cx="27" cy="27" r="16"/><path d="M39 39l13 13" stroke-width="5"/>'
            f'<path d="M18.5 31a9 9 0 0 1 17 0"/><path d="M27 31l5.5-6" stroke="var(--gold)" stroke-width="3"/>'
            f'<circle cx="27" cy="31" r="1.8" fill="currentColor"/></g>'),
    "llm": (f'<g {_S}><path d="M31 11h15l8 8v33a2 2 0 0 1-2 2H31a2 2 0 0 1-2-2V13a2 2 0 0 1 2-2z"/><path d="M46 11v8h8"/>'
            f'<path d="M35 29h13M35 35h13M35 41h13M35 47h9"/></g><path d="M8 21h19a3 3 0 0 1 3 3v12a3 3 0 0 1-3 3H17l-6 5v-5'
            f'H8a3 3 0 0 1-3-3V24a3 3 0 0 1 3-3z" fill="currentColor"/><text x="17.5" y="34" text-anchor="middle" '
            f'font-family="IBM Plex Sans,Arial,sans-serif" font-size="10" font-weight="700" fill="var(--paper)">AI</text>'),
    "report": (f'<g {_S}><path d="M16 9h22l10 10v35a2 2 0 0 1-2 2H16a2 2 0 0 1-2-2V11a2 2 0 0 1 2-2z"/>'
               f'<path d="M38 9v10h10"/><path d="M24 25h7M24 32h7M24 39h7"/><circle cx="39.5" cy="31" r="5.5"/>'
               f'<path d="M22 50v-3M27 50v-6M32 50v-4M37 50v-8M42 50v-5" stroke-width="3"/></g>'
               f'<circle cx="19.5" cy="25" r="1.7" fill="currentColor"/><circle cx="19.5" cy="32" r="1.7" '
               f'fill="currentColor"/><circle cx="19.5" cy="39" r="1.7" fill="currentColor"/>'
               f'<path d="M39.5 31v-5.5a5.5 5.5 0 0 1 5.5 5.5z" fill="var(--gold)"/>'),
    "mark": ('<rect x="2" y="2" width="8" height="8" rx="1.5" fill="currentColor"/><rect x="12" y="2" width="8" height="8" '
             'rx="1.5" fill="currentColor" opacity=".55"/><rect x="2" y="12" width="8" height="8" rx="1.5" '
             'fill="currentColor" opacity=".55"/><rect x="12" y="12" width="8" height="8" rx="1.5" fill="var(--gold)"/>'),
    "out": ('<path d="M7 17L17 7M9 7h8v8" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
            'stroke-linejoin="round"/>'),
    "drag": ('<path d="M9 7l-5 5 5 5M15 7l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2.4" '
             'stroke-linecap="round" stroke-linejoin="round"/>'),
    "next": ('<path d="M5 12h13M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
             'stroke-linejoin="round"/>'),
}


def icon(name: str, cls: str = "ic", box: int = 64) -> str:
    return f'<svg class="{cls}" viewBox="0 0 {box} {box}" aria-hidden="true" focusable="false">{ICONS[name]}</svg>'


def small(name: str) -> str:
    return f'<svg class="ic-s" viewBox="0 0 24 24" aria-hidden="true" focusable="false">{ICONS[name]}</svg>'


def external(href: str, label: str, cls: str = "") -> str:
    return (f'<a class="ext{" " + cls if cls else ""}" href="{E(href)}" target="_blank" rel="noopener">{label}'
            f'{small("out")}</a>')


# ----------------------------------------------------------------------------------------------------------------------
# Shared parts
# ----------------------------------------------------------------------------------------------------------------------

SITE_CSS = r"""
:root{--ink:#1f2430;--ink-2:#2a2f38;--muted:#5b616b;--forest-950:#0d261a;--forest-900:#123222;--forest-800:#173f2c;
  --forest-700:#22553a;--forest-600:#2e6b3a;--gold:#c9972b;--gold-deep:#8a5a14;--slate:#34485c;--paper:#f6f3ea;
  --cream:#f0f0e6;--cream-2:#e9e7da;--line:#d8d1ba;--on-forest:#efe9d6;--on-forest-2:#c3c9b5;
  --sans:'IBM Plex Sans','Helvetica Neue',Arial,sans-serif;--mono:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace}
::selection{background:#f0d9a0;color:#1f2430}
:focus-visible{outline:3px solid var(--gold);outline-offset:3px;border-radius:4px}
.ext{white-space:nowrap}
.ic-s{width:.95em;height:.95em;margin-left:.3em;vertical-align:-.12em;flex:none}
.sitebar{position:sticky;top:0;z-index:70;display:flex;align-items:center;gap:20px;min-height:56px;padding:0 24px;
  background:var(--forest-900);color:var(--on-forest);font-family:var(--sans);font-size:14px;
  box-shadow:0 1px 0 rgba(0,0,0,.25)}
.sitebar a{color:inherit;text-decoration:none}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px;letter-spacing:.01em;white-space:nowrap}
.brand svg{width:20px;height:20px;color:var(--on-forest)}
.sitebar nav{display:flex;gap:4px;margin-left:auto;overflow-x:auto;scrollbar-width:none}
.sitebar nav::-webkit-scrollbar{display:none}
.sitebar nav a{display:flex;align-items:baseline;gap:8px;padding:17px 12px 15px;border-bottom:3px solid transparent;
  white-space:nowrap;color:var(--on-forest-2);transition:color .2s,border-color .2s}
.sitebar nav a:hover{color:var(--on-forest)}
.sitebar nav a[aria-current=page]{color:var(--on-forest);border-bottom-color:var(--gold)}
.sitebar .pn{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;opacity:.8}
.sitebar .gh{display:flex;align-items:center;padding:6px 12px;border:1.5px solid rgba(239,233,214,.35);border-radius:16px;
  color:var(--on-forest);white-space:nowrap;transition:border-color .2s,background .2s}
.sitebar .gh:hover{border-color:var(--on-forest);background:rgba(239,233,214,.08)}
.tag{display:inline-flex;flex-direction:column;gap:2px;padding:8px 14px 9px;border-radius:3px;background:var(--forest-800);
  color:var(--on-forest);text-align:left;line-height:1.15}
.tag.slate{background:var(--slate)}
.tag-n{font-family:var(--mono);font-size:13px;letter-spacing:.14em;color:var(--on-forest-2)}
.tag-t{font-family:var(--sans);font-weight:700;font-size:19px;letter-spacing:-.005em}
.phasebar{background:var(--paper);border-bottom:1px solid var(--line);font-family:var(--sans);color:var(--ink)}
.pb-in{max-width:1240px;margin:0 auto;padding:30px 24px 0;display:flex;gap:24px;align-items:flex-start}
.pb-text{flex:1;min-width:0}
.pb-text h1{margin:2px 0 6px;font-size:32px;line-height:1.15;font-weight:600;letter-spacing:-.015em;text-wrap:balance}
.pb-text p{margin:0;font-size:16px;line-height:1.5;color:var(--ink-2);max-width:72ch}
.pb-text p b{font-weight:600}
.state{display:inline-flex;align-items:center;gap:8px;flex:none;margin-top:8px;padding:5px 12px;border-radius:14px;
  font-size:13.5px;font-weight:600;border:1.5px solid}
.state::before{content:"";width:8px;height:8px;border-radius:50%;background:currentColor}
.state.active{color:#7a4f0f;background:#fbefd6;border-color:#d6a54a}
.state.planned{color:#5b616b;background:#fff;border-color:#c9c6bb}
.pb-nav{max-width:1240px;margin:0 auto;padding:18px 24px 0;display:flex;flex-wrap:wrap;gap:4px 26px;font-size:14.5px}
.pb-nav a{display:inline-flex;align-items:center;padding:0 0 12px;color:var(--ink-2);text-decoration:none;
  border-bottom:2px solid transparent}
.pb-nav a:hover{color:var(--ink);border-bottom-color:var(--gold)}
.pb-nav .asof{margin-left:auto;padding-bottom:12px;color:var(--muted)}
.sitefoot{background:var(--forest-950);color:var(--on-forest-2);font-family:var(--sans);font-size:14px;line-height:1.6}
.sitefoot .in{max-width:1240px;margin:0 auto;padding:34px 24px 40px;display:flex;flex-wrap:wrap;gap:12px 40px;
  justify-content:space-between}
.sitefoot a{color:var(--on-forest);text-underline-offset:3px}
.sitefoot p{margin:0;max-width:70ch}
@media (max-width:820px){.sitebar{gap:10px;padding:0 12px}.brand span{display:none}.sitebar .pt{display:none}
  .sitebar nav a{padding:17px 9px 15px}.sitebar .gh{padding:5px 9px}
  .pb-in{flex-wrap:wrap;padding:22px 16px 0;gap:14px}.pb-text{flex:1 1 100%;order:3}.state{margin:4px 0 0 auto}
  .pb-text h1{font-size:25px}.pb-nav{padding:14px 16px 0}
  .pb-nav .asof{margin-left:0;width:100%}}
"""


def head(title: str, description: str, css: str) -> str:
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{E(title)}</title>\n'
            f'<meta name="description" content="{E(description)}">\n<meta name="theme-color" content="#123222">\n'
            f'{FONTS}\n<style>{SITE_CSS}{css}</style>\n</head>\n')


def sitebar(page: int) -> str:
    root = DEPTH[page]
    links = "".join(
        f'<a href="{root}{PAGES[n]}"{" aria-current=page" if n == page else ""}><span class="pn">Phase {n}</span>'
        f'<span class="pt">{E(p["name"])}</span></a>' for n, p in PHASES.items())
    return (f'<header class="sitebar"><a class="brand" href="{root}index.html" aria-label="Thesis home">'
            f'<svg viewBox="0 0 22 22" aria-hidden="true">{ICONS["mark"]}</svg><span>ExplainedGFM4Agri</span></a>'
            f'<nav aria-label="Thesis phases">{links}</nav>{external(REPO_URL, "GitHub", "gh")}</header>')


def tag(n: int) -> str:
    p = PHASES[n]
    return (f'<span class="tag{" slate" if p["tone"] == "slate" else ""}"><span class="tag-n">PHASE {n}</span>'
            f'<span class="tag-t">{E(p["name"])}</span></span>')


def phasebar(n: int, nav: str, asof: str = "") -> str:
    p = PHASES[n]
    rq = " ".join(f"<b>{r}</b>" for r, _ in p["rqs"])
    asof_html = f'<span class="asof">{asof}</span>' if asof else ""
    return (f'<div class="phasebar"><div class="pb-in">{tag(n)}<div class="pb-text"><h1>{E(p["topic"])}</h1>'
            f'<p>{p["so"]} · {rq} · {E(p["now"])}</p></div>'
            f'<span class="state {p["status"]}">{p["status_text"]}</span></div>'
            f'<nav class="pb-nav" aria-label="On this page">{nav}{asof_html}</nav></div>')


def footer(page: int) -> str:
    docs = " · ".join(external(f"{REPO}/docs/{d}", label) for d, label in
                      [("thesis_design.md", "thesis_design.md"), ("phase1/protocol.md", "protocol.md"),
                       ("phase1/pipeline.md", "pipeline.md")])
    return (f'<footer class="sitefoot"><div class="in"><p><b style="color:var(--on-forest)">{E(TITLE)}</b><br>'
            f'{AUTHOR} · {PROGRAMME}</p><p>Every page is built from the repository\'s documents: {docs}</p></div>'
            f'</footer>')


# ----------------------------------------------------------------------------------------------------------------------
# The landing
# ----------------------------------------------------------------------------------------------------------------------

CROP_GROUPS = [("#9cb87a", "grassland"), ("#d9a441", "winter cereals"), ("#ecd38a", "spring cereals"),
               ("#f2d42c", "rapeseed"), ("#5fa8a0", "legumes and fodder"), ("#e07b39", "maize"),
               ("#b5654a", "potatoes and vegetables")]

# The chain of phases.png: which nodes each phase holds
CHAIN = {1: [("satellite", "", ""), ("gfm", "GFM", "gold")],
         2: [("xai", "XAI · Uncertainty", "forest")],
         3: [("llm", "LLM", "slate"), ("report", "", "")]}

LANDING_CSS = r"""
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans)}
.hero{background:var(--forest-900);color:var(--on-forest);overflow:hidden}
.hero .in{max-width:1240px;margin:0 auto;padding:64px 24px 72px;display:grid;
  grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:56px;align-items:center}
.hero h1{margin:0 0 22px;font-size:clamp(30px,3.05vw,44px);line-height:1.1;font-weight:600;letter-spacing:-.02em;
  text-wrap:balance}
.hero .who{margin:0 0 4px;font-size:15.5px;color:var(--on-forest)}
.hero .who b{font-weight:600}
.hero .sup{margin:0 0 26px;font-size:14.5px;color:var(--on-forest-2)}
.hero .lead{margin:0 0 30px;font-size:17.5px;line-height:1.6;color:var(--on-forest-2);max-width:52ch}
.hero .lead b{color:var(--on-forest);font-weight:500}
.cta{display:flex;flex-wrap:wrap;gap:12px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:11px 20px;border-radius:24px;font-weight:600;font-size:15px;
  text-decoration:none;transition:background .2s,color .2s,border-color .2s,transform .2s}
.btn .ic-s{margin-left:0}
.btn.primary{background:var(--gold);color:var(--forest-950)}
.btn.primary:hover{background:#ddb04e;transform:translateY(-1px)}
.btn.ghost{border:1.5px solid rgba(239,233,214,.4);color:var(--on-forest)}
.btn.ghost:hover{border-color:var(--on-forest);background:rgba(239,233,214,.07)}
@property --x{syntax:'<percentage>';inherits:true;initial-value:56%}
.wipe{position:relative;margin:0;border-radius:6px;overflow:hidden;aspect-ratio:2/1;background:#0b1f15;
  box-shadow:0 18px 50px rgba(0,0,0,.35),0 2px 6px rgba(0,0,0,.25);--x:56%;touch-action:pan-y}
@media (prefers-reduced-motion:no-preference){.wipe.intro{animation:reveal 1.6s cubic-bezier(.16,1,.3,1) .5s backwards}}
@keyframes reveal{from{--x:97%}}
.wipe img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;image-rendering:auto}
.wipe .lab{position:absolute;inset:0;clip-path:inset(0 0 0 var(--x))}
.wipe .lab .base{filter:grayscale(1) brightness(.42) contrast(1.1)}
.wipe .bar{position:absolute;top:0;bottom:0;left:var(--x);width:2px;margin-left:-1px;background:var(--on-forest);
  box-shadow:0 0 0 1px rgba(0,0,0,.25);pointer-events:none}
.wipe .knob{position:absolute;top:50%;left:var(--x);width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;
  background:var(--on-forest);box-shadow:0 2px 8px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;
  pointer-events:none;color:var(--forest-900)}
.wipe .knob svg{width:20px;height:20px}
.wipe .cap{position:absolute;top:12px;padding:4px 10px;border-radius:3px;background:rgba(13,38,26,.78);font-size:12.5px;
  font-family:var(--mono);letter-spacing:.04em;color:var(--on-forest);pointer-events:none}
.wipe .cap.l{left:12px}.wipe .cap.r{right:12px}
.wipe input{position:absolute;inset:0;width:100%;height:100%;margin:0;opacity:0;cursor:ew-resize}
.wipe:has(input:focus-visible){outline:3px solid var(--gold);outline-offset:3px}
.figcap{margin:14px 0 0;font-size:13.5px;line-height:1.55;color:var(--on-forest-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:10px 0 0;padding:0;list-style:none;font-size:13px;
  color:var(--on-forest-2)}
.legend li{display:flex;align-items:center;gap:7px}
.legend i{width:12px;height:12px;border-radius:2px;background:var(--c)}
.phases{background:var(--cream);border-top:1px solid #d9d2bb}
.phases .in{max-width:1240px;margin:0 auto;padding:64px 24px 72px}
.phases h2{margin:0 0 8px;font-size:30px;font-weight:600;letter-spacing:-.015em}
.phases .sub{margin:0 0 40px;font-size:16.5px;line-height:1.55;color:var(--ink-2);max-width:70ch}
.chain{display:grid;grid-template-columns:repeat(3,minmax(0,1fr))}
.ph{position:relative;display:flex;flex-direction:column;padding:18px 16px 22px;border-radius:8px;color:inherit;
  text-decoration:none;transition:background .25s,box-shadow .25s}
.ph:hover,.ph:focus-visible{background:rgba(255,255,255,.6);box-shadow:0 10px 28px rgba(33,52,40,.12),0 1px 3px rgba(33,52,40,.1)}
.ph .tag{align-self:center;transition:transform .25s}
.ph:hover .tag{transform:translateY(-2px)}
.nodes{display:flex;align-items:flex-start;gap:14px;margin:26px -16px 22px;min-height:150px}
.line,.spacer{flex:1 1 0;min-width:12px}
.line{position:relative;height:2px;margin-top:51px;background:#8b9180}
.line.arr::after{content:"";position:absolute;right:-1px;top:-5px;border-top:6px solid transparent;
  border-bottom:6px solid transparent;border-left:9px solid #8b9180}
.node{display:flex;flex-direction:column;align-items:center;gap:10px;width:118px;text-align:center}
.disc{width:104px;height:104px;border-radius:50%;background:var(--cream-2);display:flex;align-items:center;
  justify-content:center;color:var(--forest-800);transition:background .25s,box-shadow .25s}
.ph:hover .disc{background:#fffdf6;box-shadow:0 0 0 3px var(--gold)}
.disc .ic{width:68px;height:68px}
.node .nl{min-height:1.4em;font-size:14px;font-weight:600;letter-spacing:.01em;color:var(--muted)}
.node .nl.gold{color:var(--gold-deep)}.node .nl.forest{color:var(--forest-700)}.node .nl.slate{color:var(--slate)}
.link{display:flex;align-items:center;width:30px;height:104px;color:#7d8472;flex:none}
.link svg{width:30px;height:16px}
.rqs{list-style:none;margin:0 0 16px;padding:0;display:flex;flex-direction:column;gap:10px;font-size:15px;line-height:1.5;
  color:var(--ink-2)}
.rqs li{display:flex;gap:10px}
.rqs b{flex:none;min-width:52px;padding:1px 0;border:1.5px solid #a9a594;border-radius:12px;text-align:center;font-size:12.5px;
  font-weight:600;color:var(--ink-2);height:fit-content}
.ph .foot{margin-top:auto;display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;padding-top:14px;
  border-top:1px solid var(--line);font-size:14px}
.ph .now{color:var(--muted);flex:1 1 180px}
.ph .go{display:inline-flex;align-items:center;gap:6px;font-weight:600;color:var(--forest-700)}
.ph .go .ic-s{margin:0;transition:transform .25s}
.ph:hover .go .ic-s{transform:translateX(3px)}
.st{font-weight:600;font-size:13px;padding:2px 10px;border-radius:12px}
.st.active{background:#fbefd6;color:#7a4f0f;border:1.5px solid #d6a54a}
.st.planned{background:#fff;color:#5b616b;border:1.5px solid #c9c6bb}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
@media (max-width:980px){.hero .in{grid-template-columns:1fr;gap:36px;padding:44px 16px 52px}
  .chain{grid-template-columns:1fr;gap:18px}.phases .in{padding:48px 16px 56px}
  .ph{padding:16px 6px 20px}.ph .tag{align-self:flex-start}.nodes{margin:18px 0 16px;min-height:0}
  .line,.spacer{display:none}}
@media (max-width:520px){.disc{width:84px;height:84px}.disc .ic{width:55px;height:55px}.node{width:96px}
  .link{height:84px;width:22px}.link svg{width:22px}.wipe .cap{font-size:11px;top:8px}.wipe .cap.l{left:8px}
  .wipe .cap.r{right:8px}.wipe .long{display:none}}
"""

LANDING_JS = r"""
const w = document.querySelector('.wipe'), r = w.querySelector('input');
const set = v => { w.classList.remove('intro'); w.style.setProperty('--x', v + '%'); r.value = v; };
r.addEventListener('input', () => set(r.value));
w.addEventListener('pointermove', e => {
  if (e.pointerType !== 'mouse') return;
  const b = w.getBoundingClientRect();
  set(Math.round(Math.max(0, Math.min(100, (e.clientX - b.left) / b.width * 100))));
});
"""


def chain_column(n: int) -> str:
    p = PHASES[n]
    nodes = []
    for k, (ic, label, tone) in enumerate(CHAIN[n]):
        if k:
            nodes.append(f'<span class="link" aria-hidden="true"><svg viewBox="0 0 30 16">{ARROW}</svg></span>')
        nodes.append(f'<span class="node"><span class="disc">{icon(ic)}</span><span class="nl {tone}">{label}</span></span>')
    nodes.insert(0, '<span class="line arr" aria-hidden="true"></span>' if n > 1 else '<span class="spacer"></span>')
    nodes.append('<span class="line" aria-hidden="true"></span>' if n < 3 else '<span class="spacer"></span>')
    rqs = "".join(f"<li><b>{r}</b><span>{E(q)}</span></li>" for r, q in p["rqs"])
    return (f'<a class="ph" href="{PAGES[n]}" aria-label="Phase {n}, {E(p["name"])}: {E(p["topic"])}">{tag(n)}'
            f'<span class="nodes">{"".join(nodes)}</span><ul class="rqs">{rqs}</ul>'
            f'<span class="foot"><span class="st {p["status"]}">{p["status_text"]}</span>'
            f'<span class="now">{E(p["now"])}</span><span class="go">Open Phase {n}{small("next")}</span></span></a>')


ARROW = ('<path d="M2 8h24M20 3l6 5-6 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
         'stroke-linejoin="round"/>')


def landing() -> str:
    legend = "".join(f'<li><i style="--c:{c}"></i>{E(t)}</li>' for c, t in CROP_GROUPS)
    knob = small("drag").replace('class="ic-s"', 'class="kn"')
    body = f"""<body>
{sitebar(0)}
<main>
<section class="hero"><div class="in">
<div>
<h1>{TITLE_HTML}</h1>
<p class="who"><b>{AUTHOR}</b> · {E(PROGRAMME)}</p>
<p class="sup">Supervisors: {E(SUPERVISORS)}</p>
<p class="lead"><b>The target system.</b> {E(TARGET)}</p>
<div class="cta"><a class="btn primary" href="{PAGES[1]}">Open Phase 1{small("next")}</a>
{external(f"{REPO}/docs/thesis_design.md", "Research design", "btn ghost")}</div>
</div>
<div>
<figure class="wipe intro">
<img src="assets/hero_s2.jpg" alt="Sentinel-2 true-colour composite of June 2021 over fields in Estonia" width="1344" height="672">
<div class="lab" aria-hidden="true"><img class="base" src="assets/hero_s2.jpg" alt=""><img src="assets/hero_labels.png" alt=""></div>
<span class="bar"></span><span class="knob">{knob}</span>
<span class="cap l">Sentinel-2<span class="long"> · June 2021</span></span><span class="cap r"><span class="long">EuroCrops </span>labels<span class="long"> · 2021</span></span>
<input type="range" min="0" max="100" value="56" aria-label="Move the line between the image and its crop labels">
</figure>
<p class="figcap">Fields in Estonia, 13.4 × 6.7 km at 10 m: eighteen chips of the Phase 1 chip set. Left of the line, the
monthly Sentinel-2 composite every model reads; right of it, the EuroCrops 2021 parcel labels the models learn to map,
shown merged into seven crop groups. Move the line to compare.</p>
<ul class="legend" aria-label="Crop groups">{legend}<li><i style="--c:#3a3f3a"></i>no label</li></ul>
</div>
</div></section>
<section class="phases" aria-labelledby="ph-h"><div class="in">
<h2 id="ph-h">Three phases</h2>
<p class="sub">From satellite time series to a report a person can check: Phase 1 maps the crops with frozen foundation
models, Phase 2 explains the maps and says how sure they are, Phase 3 writes the report from that evidence. Phase 1 is the
active workstream; Phases 2 and 3 are planned well enough to keep it compatible with them.</p>
<div class="chain">{"".join(chain_column(n) for n in PHASES)}</div>
</div></section>
</main>
{footer(0)}
<script>{LANDING_JS}</script>
</body>
</html>
"""
    return head(f"ExplainedGFM4Agri · {TITLE}", TARGET, LANDING_CSS) + body


# ----------------------------------------------------------------------------------------------------------------------
# Phases 2 and 3
# ----------------------------------------------------------------------------------------------------------------------

KIND = {"data": ("#3f6db0", "#e9f0fa", "#4a74b5"), "method": ("#8a5a14", "#fcf6ec", "#b9862a"),
        "eval": ("#2e6b3a", "#e6f0e8", "#3d8a4f")}

PHASE_CSS = r"""
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:#fff;color:var(--ink);font-family:var(--sans)}
main{max-width:1040px;margin:0 auto;padding:0 24px 72px}
main section{padding:48px 0 8px}
h2.sec{margin:0 0 10px;font-size:34px;font-weight:700;text-align:center;letter-spacing:-.01em}
.note{max-width:72ch;margin:0 auto 34px;text-align:center;font-size:16px;line-height:1.55;color:var(--muted)}
.steps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column}
.fork{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.fork .step+.step .num{margin-left:0}
.step{display:flex;gap:18px;align-items:flex-start}
.step .num{flex:none;width:42px;height:42px;border-radius:50%;color:#fff;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:18px;margin-top:14px;background:var(--c)}
.step .box{position:relative;flex:1;min-width:0;height:100%;background:var(--bg);border:2px solid var(--bd);
  border-radius:8px;padding:18px 24px 18px}
.step h3{margin:0 0 6px;font-size:21px;font-weight:600;padding-right:70px}
.step p{margin:0;font-size:16.5px;line-height:1.55;color:var(--ink-2);max-width:75ch}
.step .rq{position:absolute;right:16px;top:16px;border:2px solid var(--ink-2);border-radius:14px;padding:2px 12px;
  font-size:13.5px;font-weight:600;background:#fff}
.join{margin-left:59px;width:2px;height:30px;background:#5a6270;position:relative}
.join::after{content:"";position:absolute;bottom:-1px;left:-4px;border-left:5px solid transparent;
  border-right:5px solid transparent;border-top:8px solid #5a6270}
.rqlist{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:14px}
.rqlist li{display:flex;gap:18px;align-items:baseline;font-size:18px;line-height:1.5}
.rqlist b{flex:none;width:76px;border:2px solid var(--ink-2);border-radius:14px;padding:1px 0;text-align:center;
  font-size:15px}
.legend{display:flex;flex-wrap:wrap;gap:10px 30px;margin:26px 0 0 60px;font-size:15px;color:var(--ink-2)}
.legend span{display:flex;align-items:center;gap:9px}
.legend i{width:16px;height:16px;border-radius:3px;background:var(--bg);border:2px solid var(--bd)}
@media (max-width:700px){main{padding:0 14px 56px}h2.sec{font-size:26px}.step{gap:10px}.step .num{width:34px;height:34px;
  font-size:15px}.step .box{padding:14px 16px}.step h3{font-size:18px;padding-right:0}.step .has-rq h3{padding-top:26px}
  .fork{grid-template-columns:1fr}.join{margin-left:16px}
  .legend{margin-left:0}.rqlist li{font-size:16px;gap:10px}.rqlist b{width:64px;font-size:13px}}
"""


def plan_step(k: int, s: dict) -> str:
    c, bg, bd = KIND[s["kind"]]
    rq = f'<span class="rq">{s["rq"]}</span>' if s.get("rq") else ""
    return (f'<div class="step" style="--c:{c};--bg:{bg};--bd:{bd}"><span class="num">{k}</span>'
            f'<div class="box{" has-rq" if rq else ""}">{rq}<h3>{E(s["title"])}</h3><p>{E(s["body"])}</p></div></div>')


def phase_page(n: int) -> str:
    """A list inside PLANNED is a set of steps that run side by side on the output of the step before."""
    steps, k = [], 0
    for item in PLANNED[n]:
        group = item if isinstance(item, list) else [item]
        cells = []
        for s in group:
            k += 1
            cells.append(plan_step(k, s))
        steps.append(f'<li class="{"fork" if len(group) > 1 else "single"}">{"".join(cells)}</li>')
    legend = "".join(f'<span style="--bg:{bg};--bd:{bd}"><i></i>{t}</span>' for t, (_, bg, bd) in
                     zip(["Input", "Method", "Evaluation"], KIND.values()))
    rqs = "".join(f"<li><b>{r}</b><span>{E(q)}</span></li>" for r, q in PHASES[n]["rqs"])
    nav = ('<a href="#setup">1 · Experimental set-up</a><a href="#questions">2 · Research questions</a>'
           + external(f"{REPO}/docs/thesis_design.md", "thesis_design.md"))
    p = PHASES[n]
    body = f"""<body>
{sitebar(n)}
{phasebar(n, nav)}
<main>
<section id="setup" aria-labelledby="setup-h">
<h2 class="sec" id="setup-h">1. Experimental set-up</h2>
<p class="note">Planned, not active. The research design describes this phase only well enough to keep Phase 1 decisions
compatible with it; it is respecified once Phase 1 lands.</p>
<ol class="steps">{'<li class="join" aria-hidden="true"></li>'.join(steps)}</ol>
<div class="legend">{legend}</div>
</section>
<section id="questions" aria-labelledby="q-h">
<h2 class="sec" id="q-h">2. Research questions</h2>
<p class="note">{p["so"]}: {E(SO_TEXT[n])}</p>
<ul class="rqlist">{rqs}</ul>
</section>
</main>
{footer(n)}
</body>
</html>
"""
    return head(f"Phase {n} · {p['name']} · ExplainedGFM4Agri", f"Phase {n} of the thesis: {p['topic']}.",
                PHASE_CSS) + body


SO_TEXT = {
    1: "Label-budget characterisation of GFM-based crop segmentation against a raw-feature baseline, in-country and "
       "cross-country.",
    2: "Transparency and reliability of GFM-based crop segmentation through explainability and uncertainty "
       "quantification.",
    3: "An XAI-grounded LLM reporting pipeline producing reports traceable to the underlying evidence.",
}


def write_site() -> list[Path]:
    """Write the landing, the Phase 2 and Phase 3 pages, and .nojekyll; the Phase 1 page is build_page.py's."""
    out = []
    for n, text in [(0, landing()), (2, phase_page(2)), (3, phase_page(3))]:
        path = DOCS / PAGES[n]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        out.append(path)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    return out
