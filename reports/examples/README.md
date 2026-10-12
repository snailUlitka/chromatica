# Curated examples

One [historical original/prediction pair](published-colorization-2025.png) is
reproduced from Figure 2 on p. 1742 of the
[2025 publication](https://college.cfuv.ru/wp-content/uploads/2025/05/sbornik-czsm-2025.pdf#page=1742).
The original is on the left, model output on the right. Its
[provenance manifest](published-colorization-2025.json) records the source and
extraction; model checkpoint, source-image identity, and run configuration are
unknown. It is a published illustration, not a newly executed experiment or a
matched architecture comparison. Historical ClearML artifacts remain lost.

For future curation:

Use the same held-out images for all compared models and display lightness input,
original RGB, and predictions with consistent dimensions. Include successes and
failures and record the selection rule. Captions must identify dataset/release,
image identifier, training dataset, architecture, checkpoint SHA256, evaluation
configuration, and display gamut clipping.

Preserve source-image SHA256 and source/license attribution in a manifest. Publish
only images whose terms allow redistribution. Keep weights and bulk outputs outside
Git. Embed reviewed examples in the README after data and predictions exist.
