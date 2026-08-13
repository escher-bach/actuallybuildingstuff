# Rendering-B terminal-budget transfer correction

## Question

Does prior Rendering-A training reduce the amount of Rendering-B experience
needed to reach useful closed-loop behavior?

The earlier `step1_rendering_b_transfer_v1` run saved checkpoints from one
100M-token cosine schedule. Its intermediate checkpoints were legitimate
training-dynamics observations, but they were not terminal models optimized for
their reported token budgets. Learning rate and token exposure were therefore
confounded.

## Corrected estimand

For each arm and each nonzero Rendering-B budget, start from the arm's original
state and train an independent Hugging Face `Trainer` run whose warmup-plus-
cosine schedule terminates at that budget.

- `A-trained`: reload the exact completed Rendering-A model for every budget.
- `Init`: recreate the root-seed initialization once and reload that exact
  artifact for every budget.
- Use the same Rendering-B calibration shard and order for both arms.
- Evaluate both arms on the same held-out worlds.
- Treat budget zero only as an ungrounded-interface diagnostic.

No branch may resume from a smaller-budget endpoint. Each endpoint owns its
complete schedule and must report a terminal learning rate no greater than
`1e-9`.

## Endpoint grid and cost

| Endpoint | Updates | Nominal B tokens |
|---:|---:|---:|
| Diagnostic only | 0 | 0 |
| Terminal 3M | 92 | 3,014,656 |
| Terminal 10M | 306 | 10,027,008 |
| Terminal 30M | 916 | 30,015,488 |
| Terminal 100M | 3,052 | 100,007,936 |

The four trained endpoints total 143,065,088 nominal B tokens per arm, or
1.43 times one 100M-token run. This is intentionally smaller than independently
rerunning all twelve exploratory seed-1 checkpoints.

## Reproducibility and interpretation

The operational contract is `step1_rendering_b_terminal_transfer_v1`.
Infrastructure assertions cover source identity, calibration-prefix identity,
two-rank completion, terminal schedule ownership, numerical annealing, exact
state-dictionary serialization, aligned budgets, and matched held-out worlds.
Scientific metrics remain measurements rather than pass/fail gates.

The old continuous curves remain useful for diagnosing policy instability.
They must not be pooled with these independently annealed endpoints when
estimating sample efficiency.

Run seed 0 and seed 1 with:

- `step1/kaggle/step1_rendering_b_terminal_seed0.ipynb`
- `step1/kaggle/step1_rendering_b_terminal_seed1.ipynb`

Render each SHA-pinned notebook only after the implementation commit exists.
Attach the matching successful dense-run output to the corresponding Kaggle
notebook. Seed 2 should use the same endpoint grid after its dense source model
has completed and its exact source hashes are available.
