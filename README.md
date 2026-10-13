# Chromatica

Image colorization from CIE-Lab lightness L* to a*/b*, comparing neural
architectures and their color distributions on Food-101 and COCO.

**My contribution:** I developed the models, experiment workflows, and color
analysis tools. I coauthored the [papers](#publications) with I. V. Lezina.
— Mikhail Polevoda

![Original pizza photograph on the left and model colorization from the paper on the right](reports/examples/published-colorization-2025.png)

Original (left) and model colorization (right), from the
[CNN paper, Figure 2, p. 1742](https://college.cfuv.ru/wp-content/uploads/2025/05/sbornik-czsm-2025.pdf#page=1742).

## Results from the papers

| Experiment | Conditions | Reported MSE | Source |
| --- | --- | ---: | --- |
| Initial network without skips | Food-101 pizza only; 250 test images | ≈0.0159 | [CNN paper](https://college.cfuv.ru/wp-content/uploads/2025/05/sbornik-czsm-2025.pdf#page=1742), p. 1742 |
| Network without skips | 10 Food-101 classes; 15 epochs | 0.01313 | [U-Net paper](https://cat.gpntb.ru/?id=EC%2FShowFull&irbDb=I128F&mfn=52111), p. 283 |
| Network with skips | Same 10 classes; 15 epochs | 0.0126 | U-Net paper, p. 285 |
| Skips + simulated classifier | Same 10 classes; 15 epochs | 0.01265 | U-Net paper, p. 286 |

The values and illustration describe the published experiments, with the papers'
original precision, rather than measurements of the current revision. The U-Net
paper (p. 283) describes the earlier pizza experiment as using 200 epochs; the
CNN paper does not specify that count. The simulated classifier supplies 0.91
for the true class and 0.01 for each other class; this variant is absent from
the current package.

The [color-diversity abstract](#publications) (p. 68) reports a red-sector pixel
fraction of ≈0.269 in originals and ≈0.314 in one model's predictions (ratio ≈1.17),
using a sector about 30° wide around hue zero. It discusses Food-101 and COCO
without assigning these values to individual dataset/checkpoint pairs.

See the [published-results record](reports/published-results.md) for per-class
MSE, signed-bias ranges, and source details, or the
[standalone HTML report](reports/published-results.html).

## Architectures

| Configuration name | Historical name | Skip connections | Output activation |
| --- | --- | --- | --- |
| `encoder_decoder` | `v1`, `U_NET_WITHOUT_SKIP_CONNECTIONS` | None | Tanh |
| `unet` | `v2`, `U_NET_WITH_SKIP_CONNECTIONS` | Concatenation | ReLU |

These U-Net-like networks preserve the original convolution, batch-normalization,
and parameter-key layouts. Both accept `(N, 1, H, W)` and return `(N, 2, H, W)`;
single-image tensors are also supported. Spatial dimensions must be at least
eight and divisible by eight.

Legacy imports under `chromatica.nn.v1/v2` and raw tensor state dictionaries remain
supported. Loading a legacy dictionary requires an explicit architecture name.
Full pickled model objects are unsupported.

## Interpretation and limits

- The papers' MSE normalization/reduction and hue settings cannot be matched to
  current metrics. Historical checkpoints and raw outputs are unavailable;
  the published results do not establish current-code performance or uncertainty.
- The architectures differ in output activation as well as skip connections, so
  their MSE differences do not isolate the effect of skips. ReLU also prevents
  negative normalized a*/b* outputs.
- Hue fractions describe color distributions, not perceptual quality or diversity
  on their own. Negative prediction-minus-target Δa* denotes a greenward component
  shift; it alone cannot establish increased redness.

## Color units and metrics

The checkpoint-compatible encoding is:

```text
L_normalized  = L* / 100
ab_normalized = (ab* + 128) / 127.5 - 1
ab*           = 127.5 * ab_normalized - 0.5
```

Color analysis decodes physical a*/b* first. Signed `mean_delta_a` and
`median_delta_a` describe directional bias; `mae_ab` and `rmse_ab` measure chroma
channel errors and are not perceptual ΔE. Physical Lab errors use unclamped
predictions; RGB gamut clipping is limited to visualization.

Hue excludes zero-chroma pixels and applies the configured chroma threshold
independently to targets and predictions; reports retain both eligible counts.
Median Δa* is a histogram estimate with explicit bin width and tail counts;
undefined metrics are stored as `null`. Palette percentages describe the sampled
pixels. See the [reporting protocol](reports/README.md) for metric and comparison
requirements.

## Publications

All three papers are by М. А. Полевода and И. В. Лёзина.
Full bibliographic records are in [publications.bib](publications.bib);
cite the software separately with [CITATION.cff](CITATION.cff).

1. **Исследование применения свёрточных нейронных сетей для решения задачи
   колоризации изображений.** «Цифровые системы и модели: теория и практика
   проектирования, разработки и использования». Казань: Казанский государственный
   энергетический университет. С. 1739–1743. ISBN 978-5-89873-689-7.
   [CNN paper in the proceedings](https://college.cfuv.ru/wp-content/uploads/2025/05/sbornik-czsm-2025.pdf).
2. **Применение нейронной сети U-Net и её модификаций для решения задачи
   колоризации изображений.** «Автоматизированные системы управления и
   информационные технологии». Пермь: Издательство Пермского национального
   исследовательского политехнического университета. Т. 2. С. 282–288.
   ISBN 978-5-398-03378-6.
   [U-Net paper catalogue record](https://cat.gpntb.ru/?id=EC%2FShowFull&irbDb=I128F&mfn=52111).
3. **Исследование цветового разнообразия в изображениях и моделях колоризации.**
   XXVI Всероссийская конференция молодых учёных по математическому моделированию
   и информационным технологиям: тезисы докладов. Новосибирск: Федеральный
   исследовательский центр информационных и вычислительных технологий. С. 68.
   ISBN 978-5-905569-30-2.
   [Color-diversity abstract in the proceedings](https://expose.gpntbsib.ru/expose/elektronnye-materialy-nauchnyh-meropriyatiy-popolnenie-bazy-dannyh-nauchnye-meropriyatiya-ran-mart-2026-g.-686ea540/book/27-506801874).

## Local workflow

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Local training and offline
evaluation work with local images and checkpoints; ClearML is optional.

Install the package and development tools:

```sh
uv sync --locked --extra cpu --group dev
```

On Linux, `cpu` selects CPU-only PyTorch 2.7 wheels; `--extra cuda` selects CUDA
12.6 instead. The extras are mutually exclusive. macOS uses platform wheels.
Commands below use `--no-sync` to retain the selected dependencies. CPU is the
default device; select `mps` or `cuda` explicitly on supported hardware.
AMP is opt-in and restricted to CUDA.

Provide local images in separate training and held-out test directories:

```text
.data/food101/train/<class>/<image>.jpg
.data/food101/test/<class>/<image>.jpg
.data/coco/train/<image>.jpg
.data/coco/test/<image>.jpg
```

Images are resized to the configured square shape with bilinear antialiasing
and converted to normalized Lab. Validation is drawn only from training images
using the recorded seed. Keep official test inputs separate and preserve dataset
release/split information. Run commands from the repository root;
TOML paths are relative to the working directory.

Four [configuration templates](configs/) cover both architectures and datasets.
Their three-epoch budgets are starting points for new runs, separate from the
published protocols above. Review the device, epoch budget, and batch limits
before training. Each run creates a fresh model and refuses to overwrite an
existing output directory.

To train and evaluate all four configurations sequentially and save standalone
HTML results:

```sh
uv run --no-sync chromatica-run-all
```

This command trains four fresh models and evaluates them on the local held-out
images. It first checks split directories, image discovery, devices, and output
collisions. On a compatible Apple Silicon Mac, select the GPU with
`uv run --no-sync chromatica-run-all --device mps`; the override is recorded in
run metadata.

The final summary is `reports/index.html`; individual results are
`reports/food101_encoder_decoder.html`, `reports/food101_unet.html`,
`reports/coco_encoder_decoder.html`, and `reports/coco_unet.html`.
These self-contained reports include metric tables, hue histograms, training
history, and provenance. Checkpoints, manifests, and evaluation JSON remain in
`.runs/`; keep data, weights, caches, and bulk outputs outside Git.

To train, evaluate, and compare the two Food-101 configurations individually:

```sh
uv run --no-sync chromatica-train --config configs/food101_encoder_decoder.toml
uv run --no-sync chromatica-train --config configs/food101_unet.toml
uv run --no-sync chromatica-evaluate --config configs/food101_encoder_decoder.toml \
  --checkpoint .runs/food101_encoder_decoder/checkpoint.pth \
  --output .runs/food101_encoder_decoder/evaluation.json
uv run --no-sync chromatica-evaluate --config configs/food101_unet.toml \
  --checkpoint .runs/food101_unet/checkpoint.pth \
  --output .runs/food101_unet/evaluation.json
uv run --no-sync chromatica-compare \
  .runs/food101_encoder_decoder/evaluation.json \
  .runs/food101_unet/evaluation.json --output .runs/food101_comparison.csv
```

Repeat with COCO configurations for independent runs. Comparison requires matching
test-file hashes, image size, metric version, and evaluation settings; reports
also flag whether training protocols match. Match seeds, splits, budgets, and
preprocessing before interpreting architecture effects, and use multiple seeds
to estimate uncertainty. Batch limits bound exploratory runs; label such reports
as partial-dataset results.

## Notebooks and optional ClearML

Install notebook dependencies and start Jupyter from the repository root:

```sh
uv sync --locked --extra cpu --group notebook
scripts/lab.sh
```

The four notebooks combine research questions, protocols, and interpretation with
interactive exploration and plots. Shared computation lives in the package.
Analysis/training gates default to `False`; opening a notebook starts no experiment.

ClearML remains optional for tracking, dataset IDs, and checkpoint artifacts.
Add `--extra clearml` to the selected installation command. Configure credentials
using ClearML's user-level configuration or environment outside the repository.
Set `[tracking].enabled = true` deliberately to create a task and upload checkpoint
and metadata. For a new ClearML dataset, omit local split paths and set
`clearml_dataset_id`, `train_subdir`, and `test_subdir`. Explicit retrieval is available
through `chromatica.integrations.clearml.checkpoint_artifact(task_id, artifact_name)`.

## Development

The [CI workflow](.github/workflows/ci.yml) covers Linux CPU dependencies, Ruff,
deterministic CPU tests, and wheel build/install checks.

```sh
uv run --no-sync ruff format src tests scripts notebooks
uv run --no-sync ruff check src tests scripts notebooks
uv run --no-sync pytest tests
uv build
```

For a small CPU environment check, `uv run --no-sync chromatica-smoke` uses
synthetic 16×16 tensors and one optimization step per architecture. Tests cover
color transforms, channel pairing, metrics, short batches, datasets, palettes,
model shapes, and checkpoints.

## License and data sources

Original code is available under [MIT](LICENSE). Images and third-party materials
retain their own terms; the software license grants no dataset redistribution
rights. Obtain images from [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
and [COCO](https://cocodataset.org/#download), and review source terms before
selecting public examples.
