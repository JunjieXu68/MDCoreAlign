# MD Preprocessing

MDCoreAlign assumes that input coordinates are already suitable for structural
superposition.

For periodic MD trajectories, recommended upstream preprocessing is:

1. unwrap/nojump molecules across periodic boundaries;
2. keep protein chains whole;
3. center or otherwise consistently image the protein;
4. select a consistent atom set, commonly CA atoms for first-pass analysis;
5. verify that atom order and residue correspondence are identical across frames.

The `mdcorealign audit-preprocessing` command provides lightweight checks for
large frame-to-frame displacements and center-of-geometry drift. These checks
are not proof of correct PBC/nojump preprocessing.

Purpose-based XTC export preserves valid input unit-cell vectors after rotating
them by the same rigid transform as the coordinates. Translation is not applied
to the box. This metadata preservation does not repair wrapping or prove that
the source trajectory was made whole. Missing boxes remain missing unless the
user deliberately supplies corrected metadata upstream; MDCoreAlign never
invents a unit cell from coordinate bounds.
