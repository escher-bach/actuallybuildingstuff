# STEP 2 T4 Runbook

The sole operator entry point is:

```text
python tools/kaggle_run.py launch --experiment architecture-world-vertical-slice
python tools/kaggle_run.py status --kernel <owner/slug>
python tools/kaggle_run.py logs --kernel <owner/slug>
python tools/kaggle_run.py collect --kernel <owner/slug>
```

The launcher is generated in a temporary directory. It clones the configured
repository, checks out one exact 40-character commit, verifies that SHA, and
invokes `step2_experiments.runner` once. No model, world, test, or training
logic lives in the notebook.

The retained checkpoint remains in Kaggle output. `collect` downloads only
compact JSON evidence and logs selected by the audit pattern; it does not route
the checkpoint through the user's machine.
