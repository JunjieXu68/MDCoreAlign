# 08_xtc_export

Command:

```bash
mdcorealign apply --topology ../02_pdb_xtc_cpu/topology.pdb --trajectory ../02_pdb_xtc_cpu/trajectory.xtc --fit-selection all --apply-selection all --result-dir ../02_pdb_xtc_cpu/output --purpose trajectory --output aligned.xtc --output-dir output
```

The fixture has 5 frames and 8 CA atoms. Inspect output metadata and warnings; this is a functional smoke, not a performance benchmark.
