# Phase 1 pipeline page

`index.html` is a standalone, interactive page of the main workflow in the style of the Claude Design file
"Architecture Diagram v2": (1) the research questions above the experimental workflow and its fairness rules,
(2) the model architecture of the six arms, and (3) the six experiments of the analysis, each linked to its
`REPORT.md`, with the Estonia results report.

Online, once GitHub Pages serves `docs/` from `main` (Settings, Pages, Deploy from a branch, `main`, `/docs`):
https://davidrers.github.io/ExplainedGMF4Agri/phase1/pipeline_page/. Until then:
https://raw.githack.com/davidrers/ExplainedGMF4Agri/main/docs/phase1/pipeline_page/index.html

Interactions: **Step through** walks each diagram step by step; **Live** animates the flow along the arrows;
**Fit / 100 %** switches between the whole diagram and its design size; **Status** marks what is built in the
workflow; hovering RQ1 or a fairness rule lights the steps that carry it; clicking a model name follows its lane
and shows its settings; **Show in the architecture** points from an experiment to the part of the pipeline it
decided.

## Updating the page

The page shows what `thesis_design.md`, `protocol.md`, `pipeline.md` and the experiment reports say, and nothing
they do not. It changes with them, under the same rule: only by agreement with the user (`CLAUDE.md`).
`index.html` is generated, so never edit it by hand.

1. Edit `build_page.py`, in the place the table below names.
2. Rebuild, from the repository root (standard library only):

   ```bash
   poetry run python docs/phase1/pipeline_page/build_page.py
   ```

3. Check in a browser: open `index.html`, run Step through on both diagrams, hover RQ1 and a fairness rule,
   click each model name, toggle Status, and follow every `REPORT.md` link. Look for overlapping boxes at Fit and
   at 100 %.
4. Commit `build_page.py` and `index.html` together, staged by name, and push. Pages serves the new version a few
   minutes later.

| To change | Edit in `build_page.py` |
|---|---|
| The date in the header | `AS_OF` |
| The research questions | `RQS`, worded as in `thesis_design.md`; how the workflow answers RQ1 in `RQ1_HOW`, where it stands in `RQ1_NOW` |
| A workflow step | `WF_STEPS`: `body` is drawn, `walk` and `now` are the Step through text, `tags` the fairness rules it carries, `status` its Status chip |
| A fairness rule | `RULES` |
| A status | `met` shows Built, `partly` Partly, `no` Not yet |
| An arm's settings, shown when its name is clicked | `LANES`; what every arm shares in `SHARED` |
| The architecture drawing | Section 2 of the file. Columns sit at `X` (design pixels, each `COLW` wide), boxes are made with `box()` and arrows with `arrow()`. Each element names the arms it belongs to (`lanes`, which a model-name click lights) and its row (`row`, which a Step through step lights) |
| The architecture's Step through text | `WALKS["ar"]` |
| An experiment of the analysis | `EXPERIMENTS`: `folder` makes the `REPORT.md` link, `show` lists the ids of the architecture boxes that Show in the architecture lights |

A new arm needs a column in `X` and `NAMES`, its boxes and arrows with its key in `lanes`, an entry in `LANES`,
and a wider `STAGE_W` if the column does not fit.
