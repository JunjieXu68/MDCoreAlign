# 09_pdb_compatibility

Command:

```bash
mdcorealign apply --input trajectory.pdb --input-format ca-pdb --result-dir ../03_multimodel_pdb/output --purpose compatibility --output aligned.pdb --output-dir output
```

The fixture has 5 frames and 8 CA atoms. Inspect output metadata and warnings; this is a functional smoke, not a performance benchmark.
