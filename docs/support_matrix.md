# Support Matrix

MDCoreAlign is primarily validated for protein MD trajectories with consistent
topology and residue mapping, using C-alpha fitting as the formal default.

Other MDAnalysis-readable formats may work if topology, atom order and residue
metadata are valid, but they are not yet formally validated in the MDCoreAlign
manuscript.

Status labels:

- **Validated**: covered by the formal MDCoreAlign workflow and tests or formal benchmark systems.
- **Expected**: intended workflow when metadata and preprocessing requirements are met.
- **Experimental**: backend-readable or conceptually possible, but not a formal manuscript claim.
- **Not recommended / not formally supported**: should not be used for formal claims without separate validation.

| use_case | status | recommended_input | caveats | manuscript_claim_status |
| --- | --- | --- | --- | --- |
| Protein C-alpha MD trajectory | Validated | topology + trajectory or CA-only multi-model PDB | fixed atom order and residue metadata required | Formal default |
| Monomeric protein | Validated | PDB+XTC, GRO+XTC, or CA-only multi-model PDB | preprocess PBC/nojump/centering upstream | Formal protein workflow |
| Multichain protein | Expected | topology + trajectory with chain IDs | inspect `chain_mapping_report.tsv`; fixed chain correspondence required | Supported with caveats |
| Membrane protein | Expected | protein selection from topology + trajectory | careful PBC/centering/nojump preprocessing required; lipids not part of core inference by default | Supported with caveats |
| Protein-ligand complex | Expected | fit on protein CA; apply transform to protein or all atoms | ligand can be transformed but is not used for stable-core inference by default | Transform supported, ligand mobility not formal |
| Backbone fitting | Experimental | `--fit-selection "protein and backbone"` | not formal default; compare against CA results | Experimental |
| Heavy-atom fitting | Experimental | `--fit-selection protein` | can be slower and more sensitive to side-chain noise | Experimental |
| Coarse-grained protein | Experimental | backend-readable topology + trajectory with stable bead mapping | bead names and RMSF interpretation differ from CA atomistic workflow | Not a formal manuscript claim |
| RNA/DNA | Not recommended / not formally supported | explicit experimental atom selection only | protein CA default does not apply; no formal validation | Unsupported as formal claim |
| Protein-nucleic acid complex | Not recommended / not formally supported | analyze protein component separately where appropriate | full-complex stable-core model is not validated | Unsupported as formal full-complex claim |
| IDP/highly disordered protein | Not recommended / not formally supported | only with explicit diagnostics | stable core may not exist; weights can collapse or become reference-sensitive | Not recommended as primary use |
| Raw wrapped trajectory | Not recommended / not formally supported | preprocess before MDCoreAlign | unwrap/nojump/center upstream before interpreting RMSF | Not supported |
| Topology/trajectory atom mismatch | Not recommended / not formally supported | repair or regenerate input | fixed atom order and atom count required | Unsupported |
| Missing residues / variable atom sets | Not recommended / not formally supported | subset to common atoms first | residue correspondence must be explicit and stable | Unsupported |
| Subunit exchange or symmetry swapping | Not recommended / not formally supported | manually define fixed correspondence first | chain identity swaps invalidate ordinary residue mapping | Unsupported |

## Coordinate Export Matrix

| purpose | status | format | dependency | important limit |
| --- | --- | --- | --- | --- |
| Structure or selected models | Validated | PDBx/mmCIF | Gemmi in standard installation | text output is guarded; frame 0 is the default |
| Full aligned trajectory | Validated | XTC | MDAnalysis in standard installation | matching topology is mandatory downstream |
| Compatibility trajectory | Validated | multi-model PDB | base package | PDB field/model/text limits apply |

The compact fixture matrix validates PDB+XTC, GRO+XTC, and CA multi-model PDB
inputs for `apply`. Other MDAnalysis-readable topology/trajectory pairs may be
available but are not automatically promoted to validated status.

## Fit vs Apply Selection

- `--fit-selection` defines atoms used to estimate continuous stability weights and
  rigid-body transforms.
- `--apply-selection` defines atoms written to aligned output.
- Default `--fit-selection` is `protein and name CA`.
- Default `--apply-selection` is the same as `--fit-selection`.
- If `--apply-selection` includes atoms outside `--fit-selection`, those atoms
  are transformed but are not used for weights, RMSF, or flexible segments by
  default.

Supported PDB selection expressions are intentionally conservative:

- `protein and name CA`
- `protein and backbone`
- `protein`
- `all`

## Runtime Platform Matrix

| platform | status | evidence |
| --- | --- | --- |
| Linux x86_64, Python 3.11, CPU | Locally validated | archive-only install, full shipped tests, and I/O round trips |
| Linux x86_64, Python 3.11, CUDA 12 + CuPy 13 | Locally validated | explicit synchronized GPU smoke and CPU/GPU parity |
| Linux x86_64, Python 3.12, CPU | Locally validated | wheel install, pip check, doctor, import and toy quickstart |
| Linux x86_64, Python 3.13 | CI target, not locally validated in v6 | prepared GitHub Actions matrix; no local interpreter |
| Windows | Not validated | no passing matrix job is available |
| macOS | Not validated | no passing matrix job is available |
| CUDA 13 | Not validated | no compatible local validation was completed |

The release candidate does not claim universal OS, Python, CUDA, or
MDAnalysis-format support.
