# Bio-CBAM Revisited

**A reproducible cross-modal prior attention mechanism for facial expression recognition**

This repository contains the implementation, training pipeline, ablations, and analysis tools used for the revised Bio-CBAM study. The revision was designed as a reproducibility and falsification exercise: every reported result is tied to a frozen configuration, a stored prediction file, and a documented data-provenance record.

> **Scientific scope.** The R3 method does **not** claim an anatomical brain-to-face mapping. It does not use Thin-Plate Spline (TPS) warping in the reported experiments. Public group-level fMRI maps provide only condition-level weights, while face-space geometry comes from an independent behavioral study of diagnostic facial regions. The two coordinate systems are never treated as anatomically corresponding.

## Main result

Under the fixed CPU-budget protocol used in the manuscript, the fMRI-weighted prior did **not** significantly outperform the matched ResNet, CBAM, behavioral-prior, generic-saliency, or random-matched controls. This negative result is reported transparently. The contribution is therefore the auditable cross-modal mechanism, the matched controls, and the evidence showing the conditions under which the external prior does or does not help.

The archived study comprises **58 controlled training runs** and **15 direct FER-2013-to-JAFFE evaluations without adaptation**, producing 73 stored prediction sets.

| Evaluation protocol | Replication unit | Compared conditions |
|---|---:|---|
| FER-2013, official seven classes | 3 paired seeds | ResNet, CBAM, generic saliency, equal-weight face prior, fMRI-weighted face prior |
| FER-2013, strict four-class subset | 3 paired seeds | The five conditions above plus a trained value-distribution-matched random prior |
| JAFFE | 5 subject-disjoint folds | ResNet, CBAM, generic saliency, equal-weight face prior, fMRI-weighted face prior |
| FER-2013 → JAFFE | 15 frozen FER checkpoints | Direct evaluation without adaptation or JAFFE-based model selection |

The historical values reported in earlier manuscript drafts—94.7%, 96.3%, and 93.7%—are **not** part of this repository or the revised manuscript because their original experimental artifacts were unavailable for independent verification.

## Method

Bio-CBAM extends a ResNet backbone with channel and spatial attention after each residual stage. For stage \(b\), the external prior \(H_b\) is resized to the spatial-attention resolution and added to the learned spatial logits \(Z_b\) through an independent trainable scalar gate \(\lambda_b\):

\[
M_b = \sigma\left(Z_b + \lambda_b H_b\right),
\qquad
X'_b = X_b^c \odot M_b.
\]

The four gates are initialized to zero, optimized jointly with the network, stored in every checkpoint, and regularized with an L2 penalty. Standard ResNet and standard CBAM baselines contain no prior path. The implementation never selects a prior using the ground-truth class label.

## Public sources used by the R3 study

The cross-modal prior is constructed from two independent public sources:

| Source | Role in the method |
|---|---|
| [OpenNeuro ds003548](https://doi.org/10.18112/openneuro.ds003548.v1.0.1) and [NeuroVault collection 9492](https://identifiers.org/neurovault.collection:9492) | Public group-level emotion maps used to derive three descriptive condition weights |
| [Wegrzyn et al., 2017](https://doi.org/10.1371/journal.pone.0177239) | Published diagnostic facial-region weights defining the face-space geometry |
| FER-2013 | Training, validation, and final evaluation under the official partitions |
| JAFFE | Five-fold subject-disjoint validation and direct FER-to-JAFFE transfer |

No voxel is geometrically mapped to a face landmark, facial muscle, or action unit. The fMRI-derived quantities only calibrate the relative contribution of independently published face-region maps.

## Repository layout

| Path | Purpose |
|---|---|
| `models/bio_cbam.py` | ResNet, standard CBAM, and four-stage Bio-CBAM implementations |
| `priors/analyze_neurovault_maps.py` | Descriptive analysis of selected public group-level fMRI maps |
| `priors/build_cross_modal_prior.py` | Construction of equal-weight and fMRI-weighted functional face priors |
| `priors/analyze_threshold_sensitivity.py` | Sensitivity of fMRI weights and the resulting prior to the threshold grid |
| `priors/build_dataset_saliency_prior.py` | Training-only generic spectral-residual saliency control |
| `priors/variants.py` | Gaussian, random-matched, and generic control priors |
| `dataset_scripts/dataset_loader.py` | FER-2013 official partitions and subject-aware manifest datasets |
| `dataset_scripts/prepare_manifests.py` | JAFFE/CK+ manifest creation and subject-disjoint folds |
| `dataset_scripts/audit_dataset.py` | Pre-training split and class-distribution audit |
| `train.py` | Deterministic training, validation selection, checkpointing, resume, and final test evaluation |
| `eval.py` | Exact reconstruction and evaluation of an archived checkpoint |
| `experiments/run_ablation.py` | Matched multi-seed and multi-fold experiment orchestration |
| `experiments/analyze_runs.py` | Run aggregation, confidence intervals, paired tests, effect sizes, and Holm correction |
| `experiments/evaluate_cross_dataset.py` | Direct FER-2013-to-JAFFE evaluation without adaptation |
| `experiments/evaluate_attention_faithfulness.py` | Insertion, deletion, stability, and behavioral-region concentration metrics |
| `experiments/evaluate_prior_sensitivity.py` | Frozen-checkpoint evaluation with absent, shifted, flipped, inverted, or permuted priors |
| `experiments/analyze_calibration.py` | NLL, multiclass Brier score, fixed-bin ECE, and adaptive ECE |
| `experiments/evaluate_robustness.py` | Noise, illumination, blur, occlusion, and translation tests |
| `experiments/profile_model.py` | Parameters, FLOPs, latency, throughput, and memory profiling |
| `tests/` | Unit and end-to-end smoke tests using synthetic data only |

The files `priors/tps.py`, `priors/fmri_pipeline.py`, and `priors/generate_fmri_priors.py` are retained only as legacy research utilities. They are **not used by the R3 manuscript experiments** and must not be interpreted as part of the reported cross-modal method.

## Installation

Python 3.10–3.12 is recommended. Create an isolated environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA execution, install the PyTorch build matching the local driver before installing the remaining requirements. The published R3 campaign used a deterministic CPU protocol; GPU execution is supported but may not reproduce CPU timing measurements.

Verify the installation before using real data:

```bash
python -m unittest discover -s tests -v
python tests/smoke_training.py
```

Synthetic tests validate the software only. They are never used as scientific results.

## Data preparation

### FER-2013

The loader accepts either the standard FER-2013 CSV or the audited directory representation supported by `dataset_loader.py`. The official `Training`, `PublicTest`, and `PrivateTest` partitions are preserved. FER-2013 does not provide subject identifiers; this repository therefore makes no subject-independence claim for FER-2013.

Audit the seven-class protocol:

```bash
python -m dataset_scripts.audit_dataset \
  --dataset fer2013 \
  --data-path /ABSOLUTE/PATH/TO/FER2013 \
  --num-classes 7 \
  --output artifacts/audits/fer7.json
```

Audit the strict four-class protocol:

```bash
python -m dataset_scripts.audit_dataset \
  --dataset fer2013 \
  --data-path /ABSOLUTE/PATH/TO/FER2013 \
  --num-classes 4 \
  --four-classes angry,happy,sad,neutral \
  --output artifacts/audits/fer4.json
```

The strict protocol retains only the official `angry`, `happy`, `sad`, and `neutral` labels. No categories are merged or renamed, and the non-official label “Confusion” is never used.

### JAFFE

Create the image manifest and five subject-disjoint folds:

```bash
python -m dataset_scripts.prepare_manifests jaffe \
  --images /ABSOLUTE/PATH/TO/JAFFE \
  --output artifacts/manifests/jaffe_all.csv

python -m dataset_scripts.prepare_manifests folds \
  --manifest artifacts/manifests/jaffe_all.csv \
  --output-dir artifacts/manifests/jaffe_folds \
  --folds 5 \
  --seed 42
```

Each generated manifest contains `path`, `label`, `subject_id`, and `split`. Loading stops with an error if a subject appears in more than one partition.

> FER-2013 and JAFFE remain subject to their original distribution conditions. This repository does not redistribute their images.

## Building the public cross-modal prior

Prepare a JSON specification that maps each selected contrast label to its downloaded group-level NIfTI file. Then quantify the maps:

```bash
python -m priors.analyze_neurovault_maps \
  --spec /ABSOLUTE/PATH/selected_map_spec.json \
  --output artifacts/priors/fmri_analysis.json
```

Build the equal-weight and fMRI-weighted functional face priors from the published facial-tile table:

```bash
python -m priors.build_cross_modal_prior \
  --face-weights-csv /ABSOLUTE/PATH/weightAvgDf.csv \
  --fmri-analysis-json artifacts/priors/fmri_analysis.json \
  --output-dir artifacts/priors/cross_modal \
  --size 64
```

The output directory includes the component bank, equal-weight prior, fMRI-weighted prior, preview image, metadata, and input/output hashes. The construction is deterministic for fixed inputs.

Threshold sensitivity can be evaluated without retraining:

```bash
python -m priors.analyze_threshold_sensitivity \
  --map-spec /ABSOLUTE/PATH/selected_map_spec.json \
  --component-bank artifacts/priors/cross_modal/functional_face_components.npy \
  --thresholds 90,92.5,95,97.5,99 \
  --output artifacts/analysis/fmri_threshold_sensitivity.json
```

## Training

The manuscript’s constrained-budget FER-2013 protocol used ResNet-18, 64×64 inputs, three epochs, batch size 128, AdamW with learning rate `5e-4`, weight decay `1e-4`, dropout `0.2`, gate initialization `0`, gate regularization `1e-4`, checkpoint selection by validation macro-F1, and paired seeds `42`, `456`, and `789`.

### ResNet baseline

```bash
python train.py \
  --dataset fer2013 \
  --data-path /ABSOLUTE/PATH/TO/FER2013 \
  --output-dir runs/fer7/resnet/seed_42 \
  --num-classes 7 \
  --architecture resnet \
  --backbone resnet18 --pretrained \
  --image-size 64 --epochs 3 --batch-size 128 \
  --learning-rate 0.0005 --weight-decay 0.0001 \
  --dropout 0.2 --monitor f1_macro \
  --seed 42 --device cpu
```

### Standard CBAM baseline

Use the same command and replace:

```text
--architecture resnet
```

with:

```text
--architecture cbam
```

### Bio-CBAM with the fMRI-weighted functional prior

```bash
python train.py \
  --dataset fer2013 \
  --data-path /ABSOLUTE/PATH/TO/FER2013 \
  --output-dir runs/fer7/fmri_weighted/seed_42 \
  --num-classes 7 \
  --architecture biocbam \
  --prior artifacts/priors/cross_modal/fmri_weighted_functional_face_prior.npy \
  --require-prior \
  --backbone resnet18 --pretrained \
  --image-size 64 --epochs 3 --batch-size 128 \
  --learning-rate 0.0005 --weight-decay 0.0001 \
  --dropout 0.2 --lambda-init 0.0 \
  --lambda-regularization 0.0001 \
  --monitor f1_macro \
  --seed 42 --device cpu
```

Repeat matched runs with seeds `456` and `789`. For the strict four-class task, add:

```text
--num-classes 4 --four-classes angry,happy,sad,neutral
```

JAFFE follows the same architecture settings but uses a subject-aware fold manifest, batch size 32, ten epochs, and early-stopping patience 3.

## Matched ablations

Experiment batches are described by JSON specifications and executed with:

```bash
python experiments/run_ablation.py \
  --spec /ABSOLUTE/PATH/ablation_spec.json \
  --output-root runs/fer7
```

Validate paths and generated commands before training:

```bash
python experiments/run_ablation.py \
  --spec /ABSOLUTE/PATH/ablation_spec.json \
  --output-root runs/fer7 \
  --dry-run
```

Every matched variant must use the same partitions, architecture budget, seeds, augmentations, optimizer, and checkpoint-selection rule. A separate random-matched prior should be generated for each training seed.

## Evaluation and analysis

Evaluate a stored checkpoint:

```bash
python eval.py \
  --checkpoint runs/fer7/fmri_weighted/seed_42/best_checkpoint.pt \
  --dataset fer2013 \
  --data-path /ABSOLUTE/PATH/TO/FER2013 \
  --split test \
  --output-dir reports/fer7/fmri_weighted_seed42
```

Aggregate runs and calculate paired statistics:

```bash
python experiments/analyze_runs.py \
  --root runs/fer7 \
  --metric test_metrics.accuracy \
  --reference fmri_weighted \
  --output-dir reports/statistics/fer7
```

The R3 manuscript uses exact two-sided paired sign-flip tests because only three FER seeds or five JAFFE folds are available. Holm correction is applied within each dataset and metric family. The archived campaign analysis, not a normal-approximation t-test, is the authoritative source for manuscript values.

Direct FER-2013-to-JAFFE transfer without adaptation:

```bash
python -m experiments.evaluate_cross_dataset \
  --checkpoint-root runs/fer7 \
  --jaffe-manifest artifacts/manifests/jaffe_all.csv \
  --output-dir reports/cross_dataset \
  --device cpu
```

Calibration from archived predictions:

```bash
python -m experiments.analyze_calibration \
  --runs-root runs \
  --cross-dataset-root reports/cross_dataset \
  --output-dir reports/calibration \
  --bins 15
```

Quantitative attention faithfulness on a frozen balanced FER subset:

```bash
python -m experiments.evaluate_attention_faithfulness \
  --checkpoint-root runs/fer4 \
  --fer-data /ABSOLUTE/PATH/TO/FER2013 \
  --region-prior artifacts/priors/cross_modal/equal_weight_functional_face_prior.npy \
  --output-dir reports/attention_faithfulness \
  --per-class 64 --grid-size 8 \
  --noise-std 0.05 --subset-seed 20260902 \
  --perturbation-seed 20260902 --device cpu
```

Inference-time prior misspecification:

```bash
python -m experiments.evaluate_prior_sensitivity \
  --checkpoint-root runs/fer4 \
  --fer-data /ABSOLUTE/PATH/TO/FER2013 \
  --output-dir reports/prior_sensitivity \
  --shift-pixels 8 --block-count 16 \
  --permutation-seed 20260902 \
  --evaluation-seed 20260902 --device cpu
```

Robustness, attention export, and model profiling are available through:

```bash
python -m experiments.evaluate_robustness --help
python -m experiments.evaluate_attention --help
python -m experiments.profile_model --help
```

## Reproduced manuscript results

The values below are means from the archived R3 campaign. They are not state-of-the-art claims and should not be compared directly with studies using different architectures, resolutions, augmentations, pretraining, label spaces, or evaluation protocols.

| Dataset and protocol | ResNet | CBAM | Generic saliency | Equal-weight face prior | fMRI-weighted prior | Random-matched prior |
|---|---:|---:|---:|---:|---:|---:|
| FER-2013, official 7-class accuracy | 61.42% | **61.86%** | 60.61% | 61.65% | 61.36% | — |
| FER-2013, strict 4-class accuracy | 70.44% | 71.26% | **71.33%** | 70.85% | 70.00% | 71.20% |
| JAFFE, subject-disjoint 5-fold accuracy | 47.37% | **52.59%** | 38.93% | 43.66% | 45.64% | — |

No fMRI-weighted comparison was statistically significant after Holm correction. The learned gates remained small, and prior misspecification produced limited accuracy changes. These outcomes bound the scientific claim: the implementation is reproducible and testable, but the present data do not establish a specific performance benefit from fMRI weighting.

## Reproducibility artifacts

The companion archive `Bio-CBAM_MTAP_R3_Reproducibility_Evidence.zip` contains the exact experiment specifications, data audits, subject-disjoint manifests without images, run summaries, 73 prediction sets, paired statistics, calibration results, robustness reports, attention-faithfulness outputs, prior-sensitivity results, threshold-sensitivity results, provenance metadata, and figure-generation inputs.

Large checkpoint files and licensed datasets are intentionally excluded from the standard public archive. Every checkpoint-producing command, seed, configuration, and expected output structure is documented so that authorized users can regenerate them.

## Citation

Until the article is formally published, cite the work as a submitted manuscript and use the metadata in `CITATION.cff`. Do not assign a DOI, issue, volume, or acceptance status that has not been issued by the journal.

## License

The source code is distributed under the license in `LICENSE`. Dataset images, public neuroimaging files, and third-party behavioral supplements remain governed by their original licenses and terms of use.
