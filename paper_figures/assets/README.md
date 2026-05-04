# Paper Figure Assets

Place static, manually prepared source assets for publication figures here.

Suggested layout:

```text
paper_figures/assets/
  figure_1/
    panel_a_histology.tif
```

Use TIFF or PNG for raster image panels, and PDF or SVG for vector artwork.
Generated figure outputs should stay in `paper_figures/output/`.

For SVG assets, the figure script will use a same-stem raster export when one
exists, for example `histology.png` next to `histology.svg`.
