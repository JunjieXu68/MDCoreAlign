# Non-Protein System Policy

MDCoreAlign's formal workflow is protein C-alpha stable-core alignment and
flexibility profiling. Non-protein use is intentionally conservative.

## RNA/DNA

- Not formally supported or validated.
- The default `protein and name CA` selection does not apply.
- Experimental use requires an explicit fit selection.
- Do not output protein-specific claims such as stable protein core.
- Do not call flexible segments protein loops for nucleic acids.

## Protein-Ligand

- Expected use when fitting on protein and applying the transform to ligand or
  complex atoms.
- Ligand-specific mobility is not a formal output unless implemented
  separately.
- Ligands are not used for stable-core inference by default.

## Membrane Systems

- Supported with caveats for selected protein components.
- Require careful PBC/centering/nojump preprocessing.
- Lipids are not part of stable-core inference by default.

## Coarse-Grained Systems

- Experimental.
- Bead names and residue mapping must be checked.
- RMSF interpretation differs from atomistic C-alpha analysis.

## IDP Or Highly Disordered Systems

- Not recommended as a primary use case.
- A stable core may not exist.
- Inspect effective core size, weight distributions, convergence, and reference
  sensitivity before interpretation.
