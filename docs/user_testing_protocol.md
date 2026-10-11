# External User Testing Protocol

This protocol is for lightweight release-candidate usability testing. It is not
a benchmark and should not be used to create scientific performance claims.

## Five-Step Tester Checklist

1. Install MDCoreAlign from GitHub or TestPyPI using the command provided by the
   release manager.
2. Run `mdcorealign --help` and confirm the available commands are clear.
3. Run the toy quickstart:

   ```bash
   bash examples/toy_md/run_quickstart.sh
   ```

4. Inspect `flexibility_profile.csv` and `flexible_segments.tsv` in the output
   directory.
5. Fill in the feedback template in
   `results/final_submission_upgrade/user_testing/user_feedback_template.tsv`.

## Questions To Answer

- Did installation succeed?
- Did the CLI help make the workflow understandable?
- Did the toy quickstart finish without extra data downloads?
- Were `mobility_score`, `robust_aligned_RMSF`, and flexible segments clear?
- What was the main point of friction?
- Would you use the tool on an MD trajectory?

## Notes For Manuscript Use

Do not fabricate tester results. If no external testers have participated yet,
report the status as pending and keep this material in supplementary release
documentation.
