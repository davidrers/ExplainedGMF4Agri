# Phase 1 pipeline page

`index.html` is a standalone, interactive page of the main workflow in the style of the Claude Design file
"Architecture Diagram v2": (1) the experimental workflow with the fairness rules, (2) the model architecture of
the six arms, and (3) the six experiments of the analysis, each linked to its `REPORT.md`, with the Estonia
results report.

Online, once GitHub Pages serves `docs/` from `main`:
https://davidrers.github.io/ExplainedGMF4Agri/phase1/pipeline_page/

Interactions: **Step through** walks each diagram step by step; **Live** animates the flow along the arrows;
**Fit / 100 %** switches between the whole diagram and its design size; **Status** marks what is built in the
workflow; hovering a fairness rule lights the steps that carry it; clicking a model name follows its lane and
shows its settings; **Show in the architecture** points from an experiment to the part of the pipeline it
decided.

`index.html` is generated, so edit `build_page.py` and rebuild:

```bash
python docs/phase1/pipeline_page/build_page.py
```
