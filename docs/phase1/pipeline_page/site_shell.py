"""The site around the Phase 1 page: the landing at the site root, the Phase 2 and Phase 3 pages, and the parts every
page shares (the top bar with the three phases, the phase banner, the footer, the fonts and colours).

`build_page.py` imports this module (`site_shell`), builds the Phase 1 page with its pieces, and calls `write_site()` for the rest.
Everything the pages say comes from `docs/thesis_design.md`, `docs/phase2/explainability_setup.md` and the documents
the Phase 1 page already shows.
"""
from __future__ import annotations

import html
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/davidrers/ExplainedGMF4Agri"
REPO = f"{REPO_URL}/blob/main"
E = html.escape

TITLE = ("Towards an Explainable Panoptic Crop Mapping and Reporting System with Geospatial Foundation Models "
         "and Large Language Models")
AUTHOR = "David Reyes"
PROGRAMME = "MSc thesis, M-GEO 2026–2027, GEO-AI track · ITC, University of Twente"
SUPERVISORS = "Dr. Mahdi Farnaghi (first) and Dr. Mariana Belgiu (second)"
TITLE_HTML = E(TITLE)
TARGET = ("A user supplies an area of interest. The system returns the fields of that area with the crop each holds, "
          "explanations of why each field was mapped as it was, calibrated confidence per field, and a report written by "
          "a large language model that is traceable to that evidence.")

# Where each page sits, from the docs/ root that GitHub Pages serves
PAGES = {0: "index.html", 1: "phase1/pipeline_page/index.html", 2: "phase2/index.html", 3: "phase3/index.html"}
DEPTH = {0: "", 1: "../../", 2: "../", 3: "../"}

PHASES = {
    1: dict(name="GFMs Benchmark", topic="Panoptic crop mapping", so="SO1", status="active",
            status_text="In progress", tone="forest",
            rqs=[("RQ1.1", "How accurately can frozen geospatial foundation models, each with a trained decoder, "
                           "delineate agricultural fields and assign their crop types, compared with a segmentation "
                           "network trained on the raw imagery?"),
                 ("RQ1.2", "How does that accuracy hold as fewer fields are labelled, and across countries of the same "
                           "agro-ecological zone?")],
            now="Estonia 2021 runs end to end, fields and crops: six arms at K = 100, 20 and 5 %."),
    2: dict(name="XAI + UQ", topic="Explainability and uncertainty", so="SO2", status="planned",
            status_text="Not started", tone="forest",
            rqs=[("RQ2.1", "Can explainability methods identify the spectral and temporal evidence behind each "
                           "field's crop type and behind its borders?"),
                 ("RQ2.2", "Can uncertainty quantification give calibrated confidence for each field's crop type and "
                           "for its delineation?"),
                 ("RQ2.3", "Do the explanations and confidence agree with agronomic knowledge, such as crop calendars, "
                           "and with the raw-imagery baseline?")],
            now="Explainability set-up designed; uncertainty quantification not designed yet."),
    3: dict(name="LLM Report", topic="Evidence-grounded LLM reporting", so="SO3", status="planned",
            status_text="Not started", tone="slate",
            rqs=[("RQ3.1", "Given the field map, its explanations and its confidence as structured context, can a large "
                           "language model write agricultural reports that are faithful to that evidence?"),
                 ("RQ3.2", "Do the explanations and confidence make the reports better than the map alone, in accuracy, "
                           "traceability and usefulness?")],
            now="Planned; the first part reduced if the schedule slips."),
}

# The planned set-up of Phase 3, as docs/thesis_design.md states it. Phase 2 is XAI below.
PLANNED = {
    3: [
        dict(title="From fields to zones", kind="method",
             body="Pixel-level attribution cannot be passed to an LLM directly; the fields of the panoptic map give the "
                  "aggregation its unit, and summarising fields into zones is a Phase 3 design problem."),
        dict(title="Context document", kind="data",
             body="A structured JSON document per target zone with three blocks: predictions, explanations, and external "
                  "context (ERA5 weather variables; country, elevation and SoilGrids soil class; image-texture "
                  "descriptors where available)."),
        dict(title="Report generation", kind="method",
             body="An LLM that follows the required output format reliably and is reproducible, generating at "
                  "temperature zero and constrained to the context document."),
        dict(title="Faithfulness", kind="eval", rq="RQ3.1",
             body="Each report scored for faithfulness and traceability against its source. If explanations fail their "
                  "faithfulness checks, the context document can still operate from predictions and calibrated "
                  "uncertainty alone."),
        dict(title="What the evidence adds", kind="eval", rq="RQ3.2",
             body="Reports written from the map alone, and with its explanations and confidence, compared for "
                  "accuracy, traceability and usefulness."),
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
<p class="sub">From satellite time series to a report a person can check: Phase 1 maps the fields and their crops with
frozen foundation models, Phase 2 explains the maps and says how sure they are, Phase 3 writes the report from that
evidence. Phase 1 is the
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


# ----------------------------------------------------------------------------------------------------------------------
# Phase 2: two sections, Explainability and Uncertainty quantification. The explainability set-up is worded as
# docs/phase2/explainability_setup.md; change both together.
# ----------------------------------------------------------------------------------------------------------------------

XAI_DOC = "docs/phase2/explainability_setup.md"

XAI = dict(
    intro=("The experimental set-up for explaining the crop type of each field: what an agronomist asks of an "
           "explanation, the ground truth the explanations are judged against, and the methods that produce them. "
           "Designed, not yet run. The field borders are not explained here."),
    quote=("Agronomic experts frequently evaluate whether AI-generated explanations are biologically meaningful and "
           "consistent with established agricultural knowledge."),
    cite=("Abou Samra and Ali (2026), AgriEngineering 8(7):270", "https://doi.org/10.3390/agriengineering8070270"),
    lead="An explanation of a field's crop is useful when an agronomist can check it. Three questions do that.",
    asks=[
        dict(q="When?", sub="Crop phenology",
             ask="Does the evidence fall at the right moment of the crop's phenology?",
             points=["Every crop follows its own calendar of emergence, green-up, peak vegetation growth, flowering, "
                     "ripening and harvest.",
                     "Crops differ most around peak vegetation growth and flowering, so that is where an explanation "
                     "should fall.",
                     "In our 2021 profiles of Estonia, winter wheat and spring barley reach peak growth in June, peas in "
                     "early July and potatoes in late August."]),
        dict(q="Which property?", sub="What tells crops apart",
             ask="Does the evidence rest on a property of the plant or the soil that tells this crop apart? Each group "
                 "of bands measures one.",
             properties=True),
        dict(q="Against which crop?", sub="The crops it is confused with",
             ask="Is it the cue that separates this crop from the ones it is confused with?",
             points=["Winter wheat rather than spring wheat rests on a canopy in April.",
                     "Spring barley rather than spring wheat rests on a peak two to three weeks earlier."]),
    ],
    properties=[
        ("Greenness and canopy cover", "red, near infrared",
         "A growing crop against bare soil: winter crops are green in April, spring crops are not."),
        ("Chlorophyll", "red-edge", "Canopies that look equally green but differ in leaf chlorophyll and density."),
        ("Canopy water and dry matter", "shortwave infrared",
         "A green, water-rich canopy against a ripening one: cereals ripen in July while grassland and potatoes stay "
         "green."),
        ("Flower colour", "visible", "Rapeseed in yellow flower against cereals."),
        ("Canopy structure", "radar VH",
         "Tall, branching, broad-leaved crops against cereals: rapeseed's return in late June is far stronger than the "
         "cereals'."),
        ("Soil roughness and moisture", "radar VV",
         "Bare, tilled or ridged soil before a late crop emerges, as with potatoes."),
    ],
    truth=("What established agronomic knowledge says a model should rely on, written down per crop type and zone "
           "before any explanation is read."),
    datasets=[
        dict(title="Agro-ecological zones",
             gives="The zone of every field, so that one crop can have a different calendar in each zone.",
             cand="GAEZ v5 agro-ecological zones, already used in the transferability plan.",
             status="The zoning is still to be determined"),
        dict(title="Crop calendar maps",
             gives="Sowing, growing and harvest windows per crop and region.",
             cand="JRC ASAP crop calendar, WorldCereal crop calendars, national sources.",
             status="Coverage of Estonia and of the 20 crop types not yet checked"),
        dict(title="Theoretical band values",
             gives="The spectral and radar behaviour expected of a crop at each growth stage.",
             cand="Published Sentinel-2 and Sentinel-1 signatures per stage.",
             status="Sources to be chosen"),
    ],
    reference=("One entry per crop type, zone and growth stage: the months and the bands expected to carry the "
               "evidence, the direction, and the crop it is told apart from."),
    months=["April", "May", "June", "July", "August", "September"],
    # A cell that starts with * is a cue: what an explanation should point to
    matrix=[
        ("Winter wheat", ["*green canopy", "growth", "peak growth", "ripening", "stubble", ""],
         "Greenness in April (red, near infrared)", "spring cereals"),
        ("Spring barley", ["bare soil", "emergence", "*peak growth", "*ripening", "stubble", ""],
         "An early peak and early drying (red, near infrared, shortwave infrared)", "spring wheat and oats"),
        ("Winter rapeseed", ["green canopy", "*flowering", "*tall, branching canopy", "ripening", "stubble", ""],
         "Yellow flowers (visible); a strong radar return (VH)", "cereals"),
        ("Grassland", ["green-up", "*green", "*green", "*green, mown", "*green, mown", "*green"],
         "Never bare soil; greenness dips after mowing", "arable crops"),
        ("Peas", ["bare soil", "emergence", "*dense canopy", "peak growth, then ripening", "stubble", ""],
         "A radar return above the cereals' (VH)", "spring cereals"),
        ("Potatoes", ["bare soil", "bare, ridged soil", "emergence", "*late green-up", "*peak growth", "ripening"],
         "The latest season of all; ridged soil in spring (radar VV)", "every other spring-sown crop"),
    ],
    matrix_notes=[
        "Estonia 2021, taken as one zone, from our 2021 class profiles (580 parcels) and the monthly composites the "
        "models read.",
        "The rapeseed flowering month, the ridged potato soil and the mowing months come from general agronomy and "
        "are not yet checked for Estonia.",
        "A sample: the reference will hold every crop type, per zone, with the source of each entry.",
    ],
    assess=("The last step of the ground truth. For each crop type, zone and method, the explanation is compared with "
            "the reference."),
    assess_points=[
        "Months and bands: how much of the explanation falls in the expected months and bands, against what an even "
        "spread would give.",
        "Direction: whether the crop's score moves the way the theoretical band values say it should.",
        "Agreement with the ground truth says an explanation is agronomically plausible. Agreement between methods "
        "says it reflects the model, and not one method's behaviour.",
    ],
    experiment=("Feature attribution methods, applied to the trained models, say how much each month and band "
                "contributed to a field's crop. Their explanations go back to the assessment (2.3), round after round: "
                "what one round shows sets the next."),
    methods=[
        ("Occlusion", "Perturbation",
         "How much the field's crop score falls when a month or a band group is removed.", "Expected months and bands"),
        ("ALE, accumulated local effects", "Perturbation",
         "How a crop's score changes as the level of a band group in a month rises.",
         "Expected direction, from the theoretical band values"),
        ("SHAP", "Distillation",
         "Each month's and band group's share of the score, averaged over combinations of removed groups.",
         "Expected months and bands"),
        ("LIME", "Distillation", "The weights of a small linear model fitted around one field.",
         "Expected months and bands"),
        ("Expected or Integrated Gradients", "Backpropagation",
         "The contribution of every band and month, from the gradients of the crop score.",
         "Expected months and bands"),
    ],
    shared=[
        ("Unit", "The field. The explained quantity is the crop score averaged over the field."),
        ("Against which crop", "The score of crop A minus the score of crop B, read from the same runs."),
        ("Input groups", "Months, sensors and band groups."),
        ("A removed month", "Replaced in two ways, which show different things: interpolated from its neighbours "
                            "(events, such as a peak) and by the mean of that month over the training set (states, "
                            "such as green in April)."),
        ("Models", "The five with image inputs: TerraMind, THOR, Prithvi-EO-2.0, TESSERA v2 and the baseline trained "
                   "on the raw imagery. AlphaEarth is open: only its embedding is available, so these methods cannot "
                   "act on its inputs."),
    ],
    uq="Empty for now.",
)

# Step colours: number disc, box fill, box border, hairlines inside the box, secondary text on that fill
XAI_TONES = {
    "ask": ("#34485c", "#f2f4f6", "#7d8fa1", "#c9d2db", "#46566a"),
    "truth": ("#2e6b3a", "#e6f0e8", "#3d8a4f", "#b9d3bf", "#3f5a47"),
    "method": ("#8a5a14", "#fcf6ec", "#b9862a", "#e6d3ab", "#6b5326"),
}

XAI_CSS = r"""
.note a{color:inherit;text-underline-offset:3px}
.flow{list-style:none;margin:0;padding:0}
.part{display:flex;gap:18px;align-items:flex-start}
.part>.num{flex:none;width:42px;height:42px;border-radius:50%;color:#fff;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:18px;margin-top:18px;background:var(--c)}
.part>.box{flex:1;min-width:0;background:var(--bg);border:2px solid var(--bd);border-radius:8px;padding:22px 26px 26px}
.part h3{margin:0 0 8px;font-size:24px;font-weight:600;letter-spacing:-.01em}
.part h4{margin:30px 0 8px;font-size:18px;font-weight:600;display:flex;align-items:baseline;gap:10px}
.part h4 b{flex:none;font-weight:700;color:var(--c);font-variant-numeric:tabular-nums}
.part p{margin:0 0 12px;font-size:16.5px;line-height:1.55;color:var(--ink-2);max-width:75ch}
.pts{margin:0 0 4px;padding-left:20px;font-size:16px;line-height:1.55;color:var(--ink-2);max-width:78ch}
.pts li{margin:0 0 5px}
.pts li::marker{color:var(--c)}
.quote{margin:4px 0 22px}
.quote blockquote{margin:0;font-size:21px;line-height:1.45;font-weight:500;color:var(--ink);max-width:38em;text-wrap:balance}
.quote figcaption{margin-top:8px;font-size:14.5px;color:var(--sub)}
.quote a{color:inherit;text-underline-offset:3px}
.asks{margin:18px 0 0;display:grid;grid-template-columns:200px minmax(0,1fr);column-gap:28px}
.asks dt,.asks dd{margin:0;padding:18px 0 14px;border-top:1.5px solid var(--bd2)}
.asks dt{font-size:20px;font-weight:600;line-height:1.25}
.asks dt small{display:block;margin-top:3px;font-size:14.5px;font-weight:500;color:var(--sub)}
.tw{overflow-x:auto;margin:12px 0 10px;border:1.5px solid var(--bd2);border-radius:6px;background:#fff}
table.t{border-collapse:collapse;width:100%;font-size:14.5px;line-height:1.4;font-variant-numeric:tabular-nums}
table.t th,table.t td{padding:9px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #e3e1d6;color:var(--ink-2)}
table.t thead th{font-size:13px;font-weight:600;color:var(--sub);background:#fbfaf6;white-space:nowrap}
table.t tbody th{font-weight:600;color:var(--ink);white-space:nowrap}
table.t tbody tr:last-child>*{border-bottom:0}
table.mx th,table.mx td{padding:9px 10px}
table.mx td{min-width:84px}
table.mx td.cue{background:#fbefd6;color:#5c3d09;font-weight:600}
table.mx td.end{min-width:216px}
table.mx td.end small{display:block;margin-top:4px;font-size:13px;color:var(--sub)}
.key{display:flex;align-items:center;gap:8px;margin:0 0 10px;font-size:14.5px;color:var(--sub)}
.key i{flex:none;width:16px;height:16px;border-radius:3px;background:#fbefd6;border:1.5px solid #d6a54a}
.srcs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0 0}
.src{display:flex;flex-direction:column;background:#e9f0fa;border:2px solid #4a74b5;border-radius:8px;padding:14px 16px 12px}
.src h5{margin:0 0 6px;font-size:16.5px;font-weight:600}
.src p{margin:0 0 8px;font-size:15px;line-height:1.5;color:#22324c}
.src .cand{color:#3a4f73}
.src .stat{margin:auto 0 0;padding-top:8px;border-top:1.5px solid #b9cbe6;font-size:13.5px;font-weight:600;color:#2f4672}
.merge{width:2px;height:28px;margin:12px auto 0;background:#5a6270;position:relative}
.merge::after{content:"";position:absolute;bottom:-1px;left:-4px;border-left:5px solid transparent;
  border-right:5px solid transparent;border-top:8px solid #5a6270}
.flowfig{margin:14px 0 14px}
.flowfig .sc{overflow-x:auto;background:#fff;border:1.5px solid var(--bd2);border-radius:6px;padding:14px 12px}
.flowfig svg{display:block;width:100%;min-width:720px;height:auto}
.flowfig svg text{font-family:var(--sans)}
.flowfig figcaption{margin-top:8px;font-size:14.5px;line-height:1.5;color:var(--sub);max-width:80ch}
.shared{margin:8px 0 0;display:grid;grid-template-columns:170px minmax(0,1fr);column-gap:22px;font-size:15.5px;line-height:1.5}
.shared dt,.shared dd{margin:0;padding:9px 0;border-top:1.5px solid var(--bd2)}
.shared dt{font-weight:600}
.shared dd{color:var(--ink-2);max-width:72ch}
.empty{max-width:640px;margin:0 auto;padding:38px 24px;border:2px dashed #c9c6bb;border-radius:8px;text-align:center;
  font-size:16.5px;color:var(--muted)}
@media (max-width:700px){.part{gap:10px}.part>.num{width:34px;height:34px;font-size:15px;margin-top:14px}
  .part>.box{padding:16px 14px 18px}.part h3{font-size:20px}.quote blockquote{font-size:18px}
  .asks,.shared{grid-template-columns:minmax(0,1fr)}.asks dd,.shared dd{border-top:0;padding-top:0}
  .asks dt{padding-bottom:6px}.shared dt{padding-bottom:2px}.srcs{grid-template-columns:1fr}
  table.t{min-width:560px}table.mx{min-width:760px}table.t tbody th{white-space:normal}.quote .ext{white-space:normal}}
"""


def xai_flow() -> str:
    """The method flowchart: the assessment (last step of the ground truth), the feature attribution tree, the
    explanations, and the loop back to the assessment. Design pixels, 860 wide."""
    line, gold = "#5a6270", "#b9862a"

    def rect(x, y, w, h, fill, stroke):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="2"/>')

    def text(x, y, s, size=15.0, weight=600, fill="#1f2430", anchor="middle"):
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" '
                f'fill="{fill}">{E(s)}</text>')

    def path(d, arrow=True, stroke=line, width=2.0, marker="ah"):
        end = f' marker-end="url(#{marker})"' if arrow else ""
        return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{end}/>'

    amber, amber_bg = "#b9862a", "#fcf6ec"
    out = [
        '<defs>'
        f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
        f'<path d="M0 0L10 5L0 10z" fill="{line}"/></marker>'
        f'<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">'
        f'<path d="M0 0L10 5L0 10z" fill="{gold}"/></marker></defs>',
        # the last step of the ground truth
        rect(194, 14, 440, 70, "#e6f0e8", "#3d8a4f"),
        text(414, 43, "2.3 · Assessment, the last step of the ground truth"),
        text(414, 66, "Do the explanations agree with the agronomic reference?", 14, 400, "#2c4a35"),
        path("M414 84V124"),
        text(426, 109, "run the methods", 13, 500, "#5b616b", "start"),
        # the root
        rect(284, 126, 260, 58, "#173f2c", "#173f2c"),
        text(414, 151, "Feature attribution", 17, 600, "#ffffff"),
        text(414, 171, "on the trained models, crop head", 13, 400, "#c3c9b5"),
        path("M414 184V206", False), path("M158 206H739", False),
    ]
    # the three sub-branches and their methods: (centre, [(x, width, name lines, what it does)])
    branches = [
        ("Perturbation", 158, [(2, 150, ["Occlusion"], "one group removed"),
                               (164, 150, ["ALE"], "effect of a band level")]),
        ("Distillation", 482, [(326, 150, ["SHAP"], "shares of the score"),
                               (488, 150, ["LIME"], "local linear model")]),
        ("Backpropagation", 739, [(650, 178, ["Expected or", "Integrated Gradients"], "gradients of the score")]),
    ]
    for name, cx, methods in branches:
        out += [path(f"M{cx} 206V226"), rect(cx - 85, 228, 170, 46, amber_bg, amber), text(cx, 257, name, 16)]
        centres = [x + w / 2 for x, w, _, _ in methods]
        if len(methods) > 1:
            out += [path(f"M{cx} 274V294", False), path(f"M{centres[0]:g} 294H{centres[-1]:g}", False)]
            out += [path(f"M{c:g} 294V314") for c in centres]
        else:
            out.append(path(f"M{cx} 274V314"))
        for (x, w, lines, what), c in zip(methods, centres):
            out.append(rect(x, 316, w, 64, "#ffffff", amber))
            if len(lines) == 1:
                out += [text(f"{c:g}", 343, lines[0], 16), text(f"{c:g}", 363, what, 12.5, 400, "#5b616b")]
            else:
                out += [text(f"{c:g}", 337, lines[0], 15), text(f"{c:g}", 355, lines[1], 15),
                        text(f"{c:g}", 372, what, 12.5, 400, "#5b616b")]
            out.append(path(f"M{c:g} 380V402", False))
    out += [
        path("M77 402H739", False), path("M414 402V424"),
        # what the methods return
        rect(194, 426, 440, 66, "#e9f0fa", "#4a74b5"),
        text(414, 453, "Explanations per field and per crop", 15.5),
        text(414, 475, "which months, which bands, in which direction", 13.5, 400, "#22324c"),
        # the loop back to the assessment
        path("M634 459H832Q842 459 842 449V59Q842 49 832 49H637", True, gold, 2.5, "ag"),
        text(738, 40, "iterate", 15, 700, "#8a5a14"),
    ]
    label = ("Flowchart. The assessment, last step of the ground truth, starts feature attribution on the trained "
             "models. It has three branches: perturbation with occlusion and ALE, distillation with SHAP and LIME, and "
             "backpropagation with Expected or Integrated Gradients. The explanations they return go back to the "
             "assessment, and the loop repeats.")
    return (f'<svg viewBox="0 0 860 506" role="img" aria-label="{E(label)}" xmlns="http://www.w3.org/2000/svg">'
            f'{"".join(out)}</svg>')


def _pts(items: list[str]) -> str:
    return '<ul class="pts">' + "".join(f"<li>{E(i)}</li>" for i in items) + "</ul>"


def _table(head: list[str], rows: list[list[str]]) -> str:
    """A table whose first column is the row header."""
    th = "".join(f'<th scope="col">{E(h)}</th>' for h in head)
    body = "".join(f'<tr><th scope="row">{E(r[0])}</th>{"".join(f"<td>{E(c)}</td>" for c in r[1:])}</tr>' for r in rows)
    return f'<div class="tw"><table class="t"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _matrix() -> str:
    """The sample matrix: crop types by month, the cues marked, and what tells each crop apart."""
    th = "".join(f'<th scope="col">{E(h)}</th>' for h in ["Crop type", *XAI["months"], "What tells it apart"])
    body = []
    for crop, cells, prop, apart in XAI["matrix"]:
        tds = ""
        for c in cells:
            cue = ' class="cue"' if c.startswith("*") else ""
            tds += f"<td{cue}>{E(c.lstrip('*'))}</td>"
        body.append(f'<tr><th scope="row">{E(crop)}</th>{tds}<td class="end">{E(prop)}'
                    f'<small>Told apart from {E(apart)}</small></td></tr>')
    return (f'<div class="tw"><table class="t mx"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'
            f'</div>')


def _part(k: int, tone: str, title: str, inner: str) -> str:
    c, bg, bd, bd2, sub = XAI_TONES[tone]
    return (f'<li class="part" style="--c:{c};--bg:{bg};--bd:{bd};--bd2:{bd2};--sub:{sub}"><span class="num">{k}</span>'
            f'<div class="box"><h3>{E(title)}</h3>{inner}</div></li>')


def xai_page() -> str:
    """Phase 2: Explainability (the experimental set-up as a flow of three steps) and Uncertainty quantification."""
    x, p = XAI, PHASES[2]
    asks = []
    for a in x["asks"]:
        detail = (_table(["Property", "Bands", "How it tells crops apart"], [list(r) for r in x["properties"]])
                  if a.get("properties") else _pts(a["points"]))
        asks.append(f'<dt>{E(a["q"])}<small>{E(a["sub"])}</small></dt><dd><p>{E(a["ask"])}</p>{detail}</dd>')
    cite, href = x["cite"]
    step1 = (f'<figure class="quote"><blockquote>“{E(x["quote"])}”</blockquote>'
             f'<figcaption>{external(href, E(cite))}</figcaption></figure>'
             f'<p>{E(x["lead"])}</p><dl class="asks">{"".join(asks)}</dl>')
    srcs = "".join(f'<div class="src"><h5>{E(d["title"])}</h5><p>{E(d["gives"])}</p><p class="cand">{E(d["cand"])}</p>'
                   f'<p class="stat">{E(d["status"])}</p></div>' for d in x["datasets"])
    matrix = _matrix()
    step2 = (f'<p>{E(x["truth"])}</p>'
             f'<h4><b>2.1</b>External datasets to integrate</h4><div class="srcs">{srcs}</div>'
             f'<div class="merge" aria-hidden="true"></div>'
             f'<h4><b>2.2</b>The reference per crop type and zone</h4><p>{E(x["reference"])}</p>'
             f'<p><b>Sample matrix: the “true” cues of six crop types.</b></p>'
             f'<p class="key"><i></i>A cue: what an explanation should point to</p>{matrix}{_pts(x["matrix_notes"])}'
             f'<h4><b>2.3</b>The assessment</h4><p>{E(x["assess"])}</p>{_pts(x["assess_points"])}')
    methods = _table(["Method", "Family", "What it gives", "Compared with"], [list(m) for m in x["methods"]])
    shared = "".join(f"<dt>{E(t)}</dt><dd>{E(d)}</dd>" for t, d in x["shared"])
    step3 = (f'<p>{E(x["experiment"])}</p>'
             f'<figure class="flowfig"><div class="sc">{xai_flow()}</div><figcaption>Every method is a feature '
             f'attribution method. The gold arrow is the loop: each round of explanations returns to the assessment, '
             f'the last step of the ground truth.</figcaption></figure>'
             f'{methods}<h4>What every method shares</h4><dl class="shared">{shared}</dl>')
    flow = '<li class="join" aria-hidden="true"></li>'.join([
        _part(1, "ask", "The agronomic questions", step1),
        _part(2, "truth", "The agronomic ground truth", step2),
        _part(3, "method", "The experimentation", step3)])
    rqs = "".join(f"<li><b>{r}</b><span>{E(q)}</span></li>" for r, q in p["rqs"])
    nav = ('<a href="#explainability">1 · Explainability</a><a href="#uncertainty">2 · Uncertainty quantification</a>'
           '<a href="#questions">Research questions</a>'
           + external(f"{REPO}/{XAI_DOC}", "explainability_setup.md")
           + external(f"{REPO}/docs/thesis_design.md", "thesis_design.md"))
    body = f"""<body>
{sitebar(2)}
{phasebar(2, nav)}
<main>
<section id="explainability" aria-labelledby="xai-h">
<h2 class="sec" id="xai-h">1. Explainability</h2>
<p class="note">{E(x["intro"])}</p>
<ol class="flow">{flow}</ol>
</section>
<section id="uncertainty" aria-labelledby="uq-h">
<h2 class="sec" id="uq-h">2. Uncertainty quantification</h2>
<p class="empty">{E(x["uq"])}</p>
</section>
<section id="questions" aria-labelledby="q-h">
<h2 class="sec" id="q-h">Research questions</h2>
<p class="note">{p["so"]}: {E(SO_TEXT[2])}</p>
<ul class="rqlist">{rqs}</ul>
</section>
</main>
{footer(2)}
</body>
</html>
"""
    return head(f"Phase 2 · {p['name']} · ExplainedGFM4Agri", f"Phase 2 of the thesis: {p['topic']}.",
                PHASE_CSS + XAI_CSS) + body


SO_TEXT = {
    1: "Panoptic crop mapping with geospatial foundation models: the fields of an area and their crop types, from "
       "frozen GFMs, against a network trained on the raw imagery.",
    2: "Explaining and qualifying the panoptic map through explainability and uncertainty quantification.",
    3: "Evidence-grounded reporting with large language models, traceable to the map, its explanations and its "
       "confidence.",
}


def write_site() -> list[Path]:
    """Write the landing, the Phase 2 and Phase 3 pages, and .nojekyll; the Phase 1 page is build_page.py's."""
    out = []
    for n, text in [(0, landing()), (2, xai_page()), (3, phase_page(3))]:
        path = DOCS / PAGES[n]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        out.append(path)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    return out
