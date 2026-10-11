"""Generate the versioned literature and RMSF-label provenance audit for SE9."""

from __future__ import annotations

from common import RESULT_ROOT, write_tsv


def main() -> None:
    """Write only claims verified against primary papers or official source code."""

    output = RESULT_ROOT / "09_literature_audit"
    output.mkdir(parents=True, exist_ok=True)

    search_rows = [
        {
            "item": "mdCATH dataset and generator",
            "source_type": "primary paper and official source",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11604666/",
            "source_version": "official repository commit 71cbe9383b1911b8bc8bcce04d7c077499845f8c",
            "query_or_path": "generator/builder/molAnalyzer.py",
            "verified_finding": "Bundled RMSF is generated with MoleculeKit MetricFluctuation on CA atoms and stored in nm.",
        },
        {
            "item": "MoleculeKit RMSF implementation",
            "source_type": "versioned source release",
            "source": "https://pypi.org/project/moleculekit/1.8.36/",
            "source_version": "1.8.36",
            "query_or_path": "moleculekit/projections/metricfluctuation.py and metriccoordinate.py",
            "verified_finding": "Default trajectory alignment selection is protein CA; coordinates are centered and aligned before mean-position fluctuation.",
        },
        {
            "item": "Flexpert",
            "source_type": "primary paper and official source",
            "source": "https://github.com/KoubaPetr/Flexpert",
            "source_version": "66c5adb348b53183a1e75e7f1ad5fdb395b6ba00",
            "query_or_path": "README.md; data/scripts/extract_rmsf_labels.py",
            "verified_finding": "Supervised RMSF labels are downloaded from ATLAS and averaged over R1-R3; they are not mdCATH bundled RMSF.",
        },
        {
            "item": "DynaMode",
            "source_type": "primary paper and official source",
            "source": "https://github.com/HPuntu/DynaMode",
            "source_version": "2887c41c1292fc9efcbff04199c71914b1721b6b",
            "query_or_path": "src/dynamode/eval/evaluation.py",
            "verified_finding": "Evaluation RMSF is recomputed from generated and reference CA coordinates after full-CA Kabsch alignment to a first-frame anchor.",
        },
        {
            "item": "MarS-FM",
            "source_type": "primary paper and official source",
            "source": "https://github.com/valence-labs/mars-fm",
            "source_version": "4fc17d86cd7e22dda6353f2f1f6bf4b40c9ce946",
            "query_or_path": "scripts/analysis/analyze_mdcath.py",
            "verified_finding": "Evaluation RMSF is recomputed with MDTraj after full-atom superposition to the first reference frame and converted from nm to A.",
        },
        {
            "item": "MDLovoFit",
            "source_type": "primary paper",
            "source": "https://doi.org/10.1371/journal.pone.0119264",
            "source_version": "2015 paper; local executable 20.1.1 evaluated elsewhere",
            "query_or_path": "Methods: fractional structural fluctuation analysis",
            "verified_finding": "Iterative hard least-mobile-subset fitting reports total, least-mobile and excluded-subset RMSDs; CA RMSF is optional.",
        },
        {
            "item": "THESEUS",
            "source_type": "primary paper",
            "source": "https://doi.org/10.1093/bioinformatics/btl332",
            "source_version": "2006 paper",
            "query_or_path": "maximum-likelihood superposition model",
            "verified_finding": "Maximum-likelihood ensemble superposition models heteroscedasticity and correlations rather than selecting a fixed hard core.",
        },
        {
            "item": "Gaussian-weighted RMSD",
            "source_type": "primary paper",
            "source": "https://doi.org/10.1529/biophysj.105.066654",
            "source_version": "2006 paper",
            "query_or_path": "Gaussian-weighted RMSD fit",
            "verified_finding": "Iterative Gaussian distance weights were previously used for robust protein superposition; soft weighting alone is not unprecedented.",
        },
        {
            "item": "Robust probabilistic superposition",
            "source_type": "primary paper",
            "source": "https://doi.org/10.1186/1471-2105-11-363",
            "source_version": "2010 paper",
            "query_or_path": "heavy-tailed scale-mixture model",
            "verified_finding": "Heavy-tailed scale mixtures yield iteratively updated weighted RMSD and a probabilistic invariant-core interpretation.",
        },
        {
            "item": "THESEUS-PP",
            "source_type": "primary paper",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8515897/",
            "source_version": "2021 paper",
            "query_or_path": "Bayesian probabilistic-programming superposition",
            "verified_finding": "A Bayesian MAP formulation extends the THESEUS heteroscedastic latent-mean model.",
        },
    ]
    write_tsv(
        output / "literature_search_log.tsv",
        search_rows,
        ["item", "source_type", "source", "source_version", "query_or_path", "verified_finding"],
    )

    provenance_rows = [
        {
            "paper_or_dataset": "mdCATH bundled observable",
            "dataset": "mdCATH",
            "rmsf_used": "yes",
            "rmsf_source": "dataset generator",
            "recomputed": "generated during dataset construction",
            "alignment_convention": "MoleculeKit MetricFluctuation; default protein-CA trajectory alignment",
            "atom_selection": "CA",
            "unit": "nm",
            "role": "dataset-provided conventional observable",
            "citation": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11604666/",
        },
        {
            "paper_or_dataset": "Flexpert",
            "dataset": "ATLAS",
            "rmsf_used": "yes",
            "rmsf_source": "ATLAS analysis files, averaged over R1-R3",
            "recomputed": "no in extraction script",
            "alignment_convention": "inherited from ATLAS analysis; not established by Flexpert extraction code",
            "atom_selection": "residue-level ATLAS profile",
            "unit": "nm for model training/raw predictions per official repository note",
            "role": "training target and evaluation observable",
            "citation": "https://openreview.net/forum?id=L238BAx0wP",
        },
        {
            "paper_or_dataset": "DynaMode",
            "dataset": "mdCATH and ATLAS",
            "rmsf_used": "yes",
            "rmsf_source": "reference and generated coordinates",
            "recomputed": "yes",
            "alignment_convention": "full-CA Kabsch to first frame of first reference replicate",
            "atom_selection": "CA",
            "unit": "coordinate unit supplied to evaluation",
            "role": "evaluation metric; optional RMSF-related training terms are separately configured",
            "citation": "https://openreview.net/forum?id=0Cy0I8B9O2",
        },
        {
            "paper_or_dataset": "MarS-FM",
            "dataset": "mdCATH",
            "rmsf_used": "yes",
            "rmsf_source": "reference and generated trajectories",
            "recomputed": "yes",
            "alignment_convention": "MDTraj full-atom superposition to first reference frame before RMSF",
            "atom_selection": "matched non-hydrogen atoms for RMSF; CA used for other metrics",
            "unit": "A after explicit factor of 10",
            "role": "evaluation metric, not a bundled training label in audited code",
            "citation": "https://openreview.net/forum?id=acbf1de23fb69e87768ee9afe7d894b86854daa2",
        },
        {
            "paper_or_dataset": "DynaProt-M",
            "dataset": "ATLAS Flexpert test split",
            "rmsf_used": "yes",
            "rmsf_source": "ATLAS MD-derived profile",
            "recomputed": "not established in this audit",
            "alignment_convention": "not established in the audited paper text",
            "atom_selection": "residue-level",
            "unit": "not used for a bundled-mdCATH claim",
            "role": "evaluation target",
            "citation": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12425013/",
        },
    ]
    write_tsv(
        output / "rmsf_label_provenance.tsv",
        provenance_rows,
        [
            "paper_or_dataset",
            "dataset",
            "rmsf_used",
            "rmsf_source",
            "recomputed",
            "alignment_convention",
            "atom_selection",
            "unit",
            "role",
            "citation",
        ],
    )

    method_rows = [
        ("Ordinary Kabsch", "uniform", "least-squares", "fixed correspondence", "no", "single transform; full-set RMSD optimum", "Kabsch 1976/1978"),
        ("MDLovoFit", "hard subset", "iterative LOVO trimming", "fixed correspondence", "yes", "user-selected or scanned fraction; trajectory aggregation", "10.1371/journal.pone.0119264"),
        ("Gaussian-weighted RMSD", "continuous Gaussian", "iterative weighted least-squares", "fixed correspondence", "yes", "distance-decay parameter and initial fit affect solution", "10.1529/biophysj.105.066654"),
        ("Robust probabilistic superposition", "continuous latent scale", "EM or Gibbs", "fixed correspondence", "yes", "heavy-tailed probabilistic displacement model", "10.1186/1471-2105-11-363"),
        ("THESEUS", "inverse covariance", "maximum likelihood", "fixed correspondence/alignment supplied", "yes", "latent mean with heteroscedastic and correlated errors", "10.1093/bioinformatics/btl332"),
        ("THESEUS-PP", "Bayesian latent variance", "MAP probabilistic programming", "fixed correspondence", "yes", "Bayesian extension of THESEUS model", "PMC8515897"),
        ("MDCoreAlign", "continuous robust membership", "alternating weighted fits and robust updates", "fixed CA correspondence", "yes", "trajectory mode, optional sequence/contact regularisation; evaluated here", "this study"),
    ]
    write_tsv(
        output / "related_methods_matrix.tsv",
        [
            {
                "method": row[0],
                "weight_model": row[1],
                "optimisation": row[2],
                "correspondence": row[3],
                "iterative": row[4],
                "distinguishing_scope": row[5],
                "citation": row[6],
            }
            for row in method_rows
        ],
        ["method", "weight_model", "optimisation", "correspondence", "iterative", "distinguishing_scope", "citation"],
    )

    (output / "claim_citation_audit.md").write_text(
        "# Claim and citation audit\n\n"
        "## Safe statements\n\n"
        "- mdCATH provides a conventional CA RMSF observable generated with "
        "MoleculeKit 1.8.36 and stored in nm. The independent reproduction audit "
        "in SE5 determines whether the effective alignment convention is recovered.\n"
        "- Flexpert uses ATLAS-derived RMSF labels, not mdCATH bundled RMSF.\n"
        "- DynaMode and MarS-FM use RMSF as an evaluation observable but recompute "
        "it from coordinates under their own explicit superposition conventions.\n"
        "- Soft or probabilistic weighting for robust superposition predates "
        "MDCoreAlign; novelty must be framed around the combined trajectory workflow, "
        "fixed-budget deployment, regularisation, diagnostics and evidence package.\n\n"
        "## Statements that are not supported\n\n"
        "- Do not state that all dynamics-learning papers train on mdCATH bundled RMSF.\n"
        "- Do not treat any dataset-provided or MDCoreAlign-derived RMSF as ground truth.\n"
        "- Do not call MDCoreAlign the first continuous-weight robust superposition method.\n"
        "- Do not infer an alignment convention from the word RMSF alone.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
