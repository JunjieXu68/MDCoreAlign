# 02_pdb_xtc_cpu

Command:

```bash
mdcorealign align --topology topology.pdb --trajectory trajectory.xtc --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir output
```

The fixture has 5 frames and 8 CA atoms. Inspect output metadata and warnings; this is a functional smoke, not a performance benchmark.
