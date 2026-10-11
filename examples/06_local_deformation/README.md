# 06_local_deformation

Command:

```bash
mdcorealign local-deformation --input trajectory.pdb --input-format ca-pdb --backend cpu --block-count 2 --output-dir output
```

The fixture has 5 frames and 8 CA atoms. Inspect output metadata and warnings; this is a functional smoke, not a performance benchmark.
