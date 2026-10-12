# Reporting protocol

Historical ClearML artifacts are considered lost. No research run, notebook,
runtime test, or build was executed for this revision. Published numerical
results are transcribed in [published-results](published-results.md), with a
[standalone HTML report](published-results.html) and [structured source record](published-results.json).
They have not been revalidated with the current code. New measurements and
prediction galleries are outside this revision's scope. A historical illustration from
the 2025 publication is preserved in [examples](examples/README.md); it was extracted
without running a model. The protocol below must not be interpreted as new evidence
of model performance.

The four-run matrix below is an optional future protocol. It is not a remaining
requirement for presenting the published research, and the papers do not supply
completed reports for these configuration templates.

## Experiment matrix

| Training dataset | Architecture | Intended held-out evaluation | Current status |
| --- | --- | --- | --- |
| Food-101 | `encoder_decoder` | Food-101 test split | Unexecuted |
| Food-101 | `unet` | Food-101 test split | Unexecuted |
| COCO | `encoder_decoder` | COCO held-out split | Unexecuted |
| COCO | `unet` | COCO held-out split | Unexecuted |

Training runs must initialize independent models. Preserve dataset release,
source, split membership, image hashes, seed, image size, preprocessing,
optimization parameters, epochs, limits, environment, and device. Test data is
never a source of validation or hyperparameter selection.

Compare architectures within one training dataset first. Transfer comparisons
must identify training and evaluation datasets separately. Match split, seed,
preprocessing, optimizer, and budget; use repeated seeds and report uncertainty.
Table aggregation guards evaluation compatibility and includes training provenance,
but does not substitute for experimental review.

## Measurements

Use `physical-lab-v1` metrics and the affine inverse Lab encoding. Report MAE/RMSE
for a* and both chroma channels, normalized-ab MSE, signed mean and estimated
median delta-a*, mean chroma, hue distributions and eligible counts, red-sector
fractions, and red angular imbalance. Record bin width and histogram tails.
Undefined values are JSON `null`, not zero.

The preserved architectures differ in Tanh/ReLU output as well as skips; they
cannot isolate a skip-connection effect. Bias and hue coverage do not establish
improved diversity, perceptual quality, novelty, or SOTA. Old results cannot be
carried forward as verified measurements after correcting pairing and Lab decoding.

## Presentation

After supplying local inputs and reviewing the four TOML configurations, run
`uv run --no-sync chromatica-run-all` from the repository root in the installed
project environment. It trains and evaluates the experiment matrix sequentially,
saves one self-contained HTML per run here, and writes `index.html` after all
four evaluations and dataset-specific compatibility checks succeed. The HTML
contains metrics, hue distributions, training history, limits, and provenance;
weights and full evaluation JSON stay in `.runs/`. It does not generate mode
palettes or curate a prediction-image gallery. Existing output paths are never
overwritten. This workflow has not been executed or verified at runtime.

Keep metadata, weights, and bulk outputs in local run storage or new optional
ClearML tasks. After deliberate evaluation, select a small common set of successes
and failures for [examples](examples/README.md) and publish [figures](figures/README.md)
with provenance. Include selection rules, source licenses, model identities, and
limits. Public material must be readable without a ClearML account.
