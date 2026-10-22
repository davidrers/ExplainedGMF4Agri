# Phase 1 pipeline page

`index.html` is a standalone, interactive page of the main workflow: the experimental workflow, the
pipeline architecture of the six arms, and the six experiments of the analysis, each linked to its
`REPORT.md`. It was built on 22 October 2026 with the archify skill from `source/`.

Online, once GitHub Pages serves `docs/` from `main`:
https://davidrers.github.io/ExplainedGMF4Agri/phase1/pipeline_page/

## Rebuilding it

`source/candidate.archify.md` is the page; it reads the two diagram specs and `backbones.png` beside it.
`backbones.png` is `backbones.html` rendered by headless Chrome at a scale of 2 and a window of 1092 x 499 px.

archify refuses to write its receipts on the NFS project drive, so build on local disk and copy back:

```bash
export PATH=$HOME/.local/opt/node/bin:$PATH
cp -r docs/phase1/pipeline_page/source /tmp/pipeline_page && cd /tmp/pipeline_page
node ~/.claude/skills/archify/bin/archify.mjs finalize explainer candidate.archify.md index.html --json
cp index.html "<repository>/docs/phase1/pipeline_page/index.html"
```
