# 04_exact_audit

Command:

```bash
mdcorealign align --input trajectory.pdb --input-format ca-pdb --mode exact --backend cpu --exact-representative-count 5 --output-dir output
```

The fixture has 5 frames and 8 CA atoms. Inspect output metadata and warnings; this is a functional smoke, not a performance benchmark.
