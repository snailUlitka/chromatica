# Published results (2025)

Historical results from the author's 2025 publications. These tables are transcriptions, not a new experiment or a verification of the corrected implementation. Historical ClearML datasets, weights, and raw outputs are lost. No training, evaluation, notebook execution, runtime tests, or build was performed for this revision. New experiments are outside its scope.

## Published MSE summary

| Published model | Data scope | Reported MSE | Printed page |
| --- | --- | --- | --- |
| Initial network without skips | Food-101 pizza-only experiment | approximately 0.0159 | 1742 |
| Network 1: without skips | Food-101, 10 classes | 0.01313 | 283 |
| Network 2: with skips | Food-101, 10 classes | 0.0126 | 285 |
| Network 3: skips + simulated classifier | Food-101, 10 classes | 0.01265 | 286 |

The April paper reports the approximately 0.0159 MSE on 250 test images (p. 1742); its training description gives a pizza-only initial stage (p. 1741). The June U-Net paper describes the earlier experiment as pizza-only training and testing, with 200 epochs (p. 283). The April paper itself does not state the epoch count. The exact MSE scaling/reduction and checkpoint identities remain unavailable.

## Per-class MSE (U-Net paper)

| Food-101 class | Without skips | With skips | Simulated classifier |
| --- | --- | --- | --- |
| greek_salad | 0.01556 | 0.01541 | 0.01523 |
| risotto | 0.01041 | 0.01078 | 0.01136 |
| pizza | 0.01557 | 0.01455 | 0.01508 |
| apple_pie | 0.01251 | 0.01124 | 0.01173 |
| pancakes | 0.01171 | 0.01131 | 0.01212 |
| lasagna | 0.01287 | 0.01235 | 0.01206 |
| creme_brulee | 0.01374 | 0.01287 | 0.01316 |
| waffles | 0.01180 | 0.01153 | 0.01206 |
| pho | 0.01426 | 0.01388 | 0.01139 |
| spaghetti_bolognese | 0.01285 | 0.01207 | 0.01227 |

Pages 283 (without skips), 284–285 (with skips), and 285–286 (simulated classifier). All three use ten classes and 15 epochs. The simulated classifier supplies 0.91 for the true class and 0.01 for each other class. Published means and precision are retained; no averages or significance estimates are recomputed here.

## Color-diversity abstract (p. 68)

| Quantity | Value as published |
| --- | --- |
| Original red-sector pixel fraction | approximately 0.269 |
| Prediction red-sector pixel fraction (one model) | approximately 0.314 |
| Prediction minus original red-sector fraction | approximately 0.045 |
| Prediction/original red-sector fraction ratio | approximately 1.17 |
| Mean signed delta-a* range | -1.29 to -0.97 |
| Median signed delta-a* range | -0.54 to -0.32 |

The abstract discusses Food-101 and COCO, but does not assign these values to individual dataset/checkpoint pairs. Its red wedge is described as about 30 degrees wide around hue zero. Its stated conclusion is that a warm input bias is reproduced and amplified by the models. Treat this as a published interpretation, subject to the limitations below. Negative prediction-minus-target delta-a* alone is a greenward component shift and cannot establish an increase in redness.

## Provenance and limitations

- Values are transcribed from publications, not outputs of the current code; retain the published precision and averages.
- Historical datasets, checkpoints, seeds, manifests, per-run attribution, and detailed reductions cannot be recovered.
- MSE uses the publication's Lab representation; its exact normalization and reduction have not been established for comparison with current normalized_mse or physical Lab metrics.
- The third network uses simulated class information (0.91 for the true class, 0.01 for each other class), not a learned classifier; it is not implemented in the current package.
- The abstract discusses both Food-101 and COCO but does not assign each numerical example to a dataset/checkpoint; do not populate the current four-run matrix from these values.
- The abstract describes a red wedge about 30 degrees wide around zero; its thresholds and width convention cannot be matched to the current metric settings.
- Negative signed delta-a* under prediction-minus-target denotes a greenward component shift; it alone does not establish increased redness. The abstract's warm-bias interpretation also uses hue fractions.
- Historical hue pairing and Lab decoding are not revalidated after the current fixes. These numbers cannot be treated as physical-lab-v1 measurements.
- The preserved architectures differ in output activation as well as skip connections; reported MSE differences do not isolate a skip-connection effect.
- No standard deviations, confidence intervals, current MAE/RMSE, or new results are inferred from the papers.

## Sources

- [Исследование применения свёрточных нейронных сетей для решения задачи колоризации изображений](https://college.cfuv.ru/wp-content/uploads/2025/05/sbornik-czsm-2025.pdf); SHA256 of the inspected PDF: `a621fb9331e6a9ce1926fc0b220d823a3e08660265a96351cf2ef6e2fb1c7e56`.
- [Применение нейронной сети U-Net и её модификаций для решения задачи колоризации изображений](https://cat.gpntb.ru/?id=EC%2FShowFull&irbDb=I128F&mfn=52111); SHA256 of the inspected PDF: `5a99628a8ec121342f7172f4621c169e221fc5ecfab10962ce426a6ee06ba834`.
- [Исследование цветового разнообразия в изображениях и моделях колоризации](https://expose.gpntbsib.ru/expose/elektronnye-materialy-nauchnyh-meropriyatiy-popolnenie-bazy-dannyh-nauchnye-meropriyatiya-ran-mart-2026-g.-686ea540/book/27-506801874); SHA256 of the inspected PDF: `ccd85639c9f0d5dd5447fda92acccc800307aac486125df128d6499dd4a6d574`.

The PDF hashes identify the inspected source files, including the author's
excerpt attachments; they are not hashes of recovered experiment artifacts.
Numbers and prose were checked against rendered pages. No source PDFs are
redistributed here. The [JSON transcription](published-results.json) has
`kind = historical_publication_transcription`, not an `evaluated` report;
it must not be fed into the current evaluation-comparison CLI.
