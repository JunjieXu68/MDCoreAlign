# Alignment-Free Local Deformation

`mdcorealign local-deformation` calculates a chain-aware W5 pair-distance
instability signal. Pair distances make the descriptor invariant to global
rotation and translation; its scale-normalised formulation is dimensionless.

Outputs include local-deformation index, local-rigidity score, contiguous
locally deforming segments, confidence and temporal-block reproducibility.
These complement stable-core alignment:

| Global displacement | Local deformation | Conservative interpretation |
| --- | --- | --- |
| Low | Low | Globally and locally stable |
| High | Low | Coherently displaced geometric region |
| High | High | Global displacement with internal deformation |
| Low | High | Local strain or transient local event |

Do not call W5 RMSD, global mobility or a mature hard biological-domain label.
High-displacement/low-deformation regions require external annotation before
being described as biological domains.

For the locked presentation case studies, high means `>= 0.70`, low means
`<= 0.30`, and values strictly between 0.30 and 0.70 remain intermediate.
These display thresholds do not convert W5 into a biological annotation.
