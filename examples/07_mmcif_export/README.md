# 07_mmcif_export

Command:

```bash
mdcorealign apply --input trajectory.pdb --input-format ca-pdb --result-dir ../03_multimodel_pdb/output --purpose structure --frames 0,2,4 --output aligned.cif --output-dir output
```

The fixture has 5 frames and 8 CA atoms. Inspect output metadata and warnings; this is a functional smoke, not a performance benchmark.
