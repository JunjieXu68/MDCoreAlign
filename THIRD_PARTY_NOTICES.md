# Third-party notices

MDCoreAlign does not redistribute the following third-party projects.

- **NumPy** provides arrays and CPU numerical operations. It is a standard
  runtime dependency; see <https://numpy.org/> and its BSD licence.
- **SciPy** provides scientific numerical routines. It is a standard runtime
  dependency; the validated 1.16.3 distribution declares BSD-3-Clause.
  See <https://scipy.org/> and the licenses included in the installed archive.
- **MDAnalysis** provides validated topology/trajectory reading and XTC
  writing. It is a standard runtime dependency; see
  <https://www.mdanalysis.org/>. The validated MDAnalysis 2.10.0 archive declares
  LGPL-3.0-or-later; consult the installed version's own license, as historical
  distributions need not use the same license declaration.
- **Gemmi** provides validated PDBx/mmCIF writing and parsing. It is a standard
  runtime dependency; see <https://gemmi.readthedocs.io/> and its Mozilla
  Public License 2.0.
- **CuPy** provides the optional CUDA execution backend. It is not required for
  CPU use and must match the local CUDA environment; see
  <https://cupy.dev/> and its MIT licence.
- **MDLovoFit** and **THESEUS** were external research comparators. Their source
  and binaries are not bundled in MDCoreAlign.

These libraries do not alter MDCoreAlign's registered alignment mathematics;
MDAnalysis and Gemmi complete the standard input/output product surface.
