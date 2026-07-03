# Engineering Report

## Problem

The goal is to classify an input image as a genuine photo or a screen-recaptured fake photo. The hard part is that the two classes can be visually similar at the semantic level, while the discriminative signal lives in local texture, frequency structure, and blur characteristics.

## Dataset

The repository uses a small image dataset organized into `dataset/real` and `dataset/fake`. The pipeline validates files, skips unreadable images, removes exact MD5 duplicates, and performs a stratified 70/15/15 train/validation/test split.

## Design Decisions

The system is intentionally classical rather than deep-learning based. That choice is pragmatic for the deployment target: the image evidence is mostly high-frequency and local, the model is compact, and inference is fully CPU based. The result is predictable latency and no GPU or cloud dependency.

## Patch Strategy

Each image is converted into five fixed $256 \times 256$ patches from the four corners and the center. Reflection padding is used when the image is smaller than the patch size. This keeps the local evidence intact and avoids relying on one crop or on resized global context.

## Why Classical CV

Classical CV is a good fit here because the strongest signal is not object identity but physical capture artifacts. FFT, LBP, Laplacian, HSV, and Canny features encode those artifacts directly. That makes the pipeline smaller, faster, and easier to reason about than a heavier neural baseline.

## Feature Engineering

The patch-level feature set is built from five families:

- FFT: captures periodic screen-grid artifacts, spectral spikes, anisotropy, and high-frequency concentration.
- LBP: captures local micro-texture patterns that appear in recaptured imagery.
- Laplacian: captures sharpness and edge attenuation.
- HSV: captures color variability while reducing dependence on raw scene content.
- Canny: captures edge density as a coarse structural descriptor.

DoG and CLAHE are used as preprocessing steps. DoG suppresses smooth low-frequency illumination while emphasizing fine detail; CLAHE reduces local exposure variation before texture and color statistics are measured.

## Model Comparison

The final predictor is a soft-voting ensemble of LightGBM and XGBoost. Both models are well suited to tabular features and perform strongly on the handcrafted representation. The ensemble improves robustness relative to a single tree-based model while keeping the inference footprint small.

## Cross-Validation

Model selection uses stratified 5-fold cross-validation. A small hyperparameter grid is searched for each base learner, then the ensemble weights are tuned on validation folds. This helps keep the final configuration stable across folds instead of depending on a single split.

## Threshold Optimization

The final decision threshold is optimized on validation probabilities using a fine-grained search. This is important because the output is a fake probability, and the best classification threshold is not necessarily $0.5$.

## Latency and Cost

Measured sequential inference latency is 87.94 ms/image on CPU, and the serialized model is 216.11 KB. Inference runs entirely on-device, so cloud inference cost is approximately $0.

## Failure Cases

The main failure mode is aggressively compressed or low-resolution recaptures. When the source image is heavily degraded, the moiré and screen-grid artifacts can become too weak to separate from natural sensor noise.

## Limitations

The approach depends on local texture quality. Images with severe compression, resizing, or cropping can reduce the signal. The system is also tuned to the dataset distribution in this repository, so performance may shift under different capture conditions.

## Future Improvements

The most useful next step would be broader capture-condition coverage: more low-quality recaptures, more device variation, and more near-duplicate control. If compute budgets change, a lightweight learned texture model on top of the current preprocessing stack would be the natural follow-up.
