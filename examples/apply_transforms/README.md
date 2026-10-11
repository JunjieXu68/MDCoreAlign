# Save and Apply Transforms

Run `bash examples/apply_transforms/run.sh`. The first command estimates weights
and explicitly fits all frames without writing aligned coordinates. The second
validates and applies `transforms.npz`; it does not rerun the alignment
algorithm. PDB is the stable materialized output format in this release.
