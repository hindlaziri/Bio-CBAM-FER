# Bio-CBAM — Code expérimental révisé

Ce dépôt fournit une implémentation reproductible de **Bio-CBAM**, où un prior spatial externe est combiné aux logits d’attention spatiale CBAM après chacun des quatre stages d’un backbone ResNet. La fusion suit

\[
M_b = \sigma(Z_b + \lambda_b H_b), \qquad F'_b = F_b \odot M_b,
\]

où chaque $\lambda_b$ est appris et enregistré dans les checkpoints.

> **État scientifique.** Le package ne contient ni données humaines, ni cartes fMRI, ni résultats pré-calculés. Il ne revendique donc aucun score avant exécution sur les données réelles. Les tableaux du manuscrit doivent être remplis uniquement à partir des fichiers JSON produits par ces scripts.

## Composants

| Dossier ou fichier | Fonction |
|---|---|
| `models/bio_cbam.py` | ResNet-18/50 multi-échelle, quatre Bio-CBAM, portes $\lambda_b$ et mélange de priors sans utilisation de l’étiquette vraie |
| `priors/fmri_pipeline.py` | Chargement des cartes statistiques, projection 2D documentée, normalisation et TPS à partir de correspondances explicites |
| `priors/variants.py` | Priors de contrôle aléatoire apparié, gaussien central et salience spectrale générique |
| `dataset_scripts/dataset_loader.py` | Splits officiels FER-2013, manifests par sujet pour CK+/JAFFE et audit SSIM |
| `dataset_scripts/prepare_manifests.py` | Création des manifests CK+/JAFFE et folds strictement séparés par sujet |
| `train.py` | Entraînement, validation, checkpointing, reprise et évaluation finale unique du test |
| `eval.py` | Reconstruction exacte et évaluation d’un checkpoint |
| `experiments/` | Ablations, statistiques, McNemar, robustesse, attention, complexité et tableaux LaTeX |
| `tests/` | Tests logiciels synthétiques, jamais utilisés comme résultats scientifiques |

## Installation

Une version moderne de Python 3.10–3.12 est recommandée. Créez un environnement isolé, puis installez les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour une installation CUDA, installez d’abord la version de PyTorch compatible avec votre pilote depuis le sélecteur officiel, puis exécutez la dernière commande.

## 1. Construction d’un prior fMRI/TPS

Le pipeline exige une carte d’activation réelle et un CSV contenant `source_x,source_y,target_x,target_y`. Ces correspondances constituent une hypothèse expérimentale documentée, et non une projection anatomique cerveau–muscle.

```bash
python -m priors.generate_fmri_priors \
  --source /ABSOLUTE/PATH/group_stat_map.npy \
  --correspondences /ABSOLUTE/PATH/correspondences.csv \
  --output-stem artifacts/priors/fmri \
  --height 224 --width 224 \
  --activation-mode positive \
  --threshold-percentile 95 \
  --smooth-sigma 2.0 \
  --tps-regularization 0.001
```

L’exécution produit `.npy`, `.png` et `.json`, avec checksums des entrées et de la sortie. Voir `priors/README.md`.

## 2. Priors de contrôle

```bash
python -m priors.variants random_matched \
  --reference artifacts/priors/fmri.npy --seed 42 \
  --output-stem artifacts/priors/random_matched_seed42

python -m priors.variants gaussian_center \
  --height 224 --width 224 --sigma-fraction 0.2 \
  --output-stem artifacts/priors/gaussian_center

python -m priors.variants generic_saliency \
  --fer-csv /ABSOLUTE/PATH/fer2013.csv \
  --output-stem artifacts/priors/generic_saliency
```

La salience générique est la moyenne, sur le training set uniquement, des cartes obtenues par l’algorithme spectral residual. L’algorithme et le nombre d’images sont enregistrés dans les métadonnées.

## 3. FER-2013

Le loader utilise exclusivement `Training`, `PublicTest` et `PrivateTest`. FER-2013 ne fournit pas d’identifiants de sujets; le code ne qualifie donc pas ce protocole de subject-independent.

### Sept classes officielles

```bash
python train.py \
  --dataset fer2013 \
  --data-path /ABSOLUTE/PATH/fer2013.csv \
  --output-dir runs/fer7/fmri/seed_42 \
  --num-classes 7 \
  --prior artifacts/priors/fmri.npy --require-prior \
  --backbone resnet50 --pretrained \
  --epochs 100 --batch-size 32 --seed 42 --amp
```

### Sous-ensemble strict de quatre classes

Le réglage par défaut conserve uniquement `angry,happy,sad,neutral`. Il ne fusionne aucune classe et refuse explicitement l’étiquette non officielle « Confusion ».

```bash
python train.py \
  --dataset fer2013 \
  --data-path /ABSOLUTE/PATH/fer2013.csv \
  --output-dir runs/fer4/fmri/seed_42 \
  --num-classes 4 \
  --four-classes angry,happy,sad,neutral \
  --prior artifacts/priors/fmri.npy --require-prior \
  --backbone resnet50 --pretrained --seed 42
```

Le filtrage SSIM est optionnel et ne retire que des quasi-doublons du training set. Il produit `ssim_training_audit.json`. Les partitions de validation et de test ne sont jamais modifiées.

## 4. CK+ et JAFFE avec séparation par sujet

Créez d’abord un manifest sans colonne `split`, puis générez les folds. Pour JAFFE :

```bash
python -m dataset_scripts.prepare_manifests jaffe \
  --images /ABSOLUTE/PATH/jaffe \
  --output manifests/jaffe_all.csv

python -m dataset_scripts.prepare_manifests folds \
  --manifest manifests/jaffe_all.csv \
  --output-dir manifests/jaffe_folds --folds 5 --seed 42
```

Pour CK+ :

```bash
python -m dataset_scripts.prepare_manifests ckplus \
  --images /ABSOLUTE/PATH/cohn-kanade-images \
  --emotion-labels /ABSOLUTE/PATH/Emotion \
  --output manifests/ckplus_all.csv

python -m dataset_scripts.prepare_manifests folds \
  --manifest manifests/ckplus_all.csv \
  --output-dir manifests/ckplus_folds --folds 5 --seed 42
```

Chaque manifest final contient `path`, `label`, `subject_id` et `split`. Le loader s’arrête si un sujet apparaît dans plusieurs partitions.

## 5. Ablations multi-runs

Copiez `experiments/ablation_spec_TEMPLATE.json`, remplacez tous les chemins, puis lancez :

```bash
python experiments/run_ablation.py \
  --spec experiments/ablation_spec.json \
  --output-root runs/fer7_ablation
```

Chaque variante utilise les mêmes seeds et hyperparamètres. `resnet` constitue la baseline sans attention; `cbam` applique l’attention standard sans prior; les autres variantes utilisent Bio-CBAM avec le prior indiqué.

## 6. Statistiques

```bash
python experiments/analyze_runs.py \
  --root runs/fer7_ablation \
  --metric test_metrics.accuracy \
  --reference fmri \
  --output-dir reports/statistics
```

Lorsque les seeds correspondent exactement, l’outil applique un test t apparié bilatéral et rapporte Cohen $d_z$. Dans le cas contraire, il utilise le test de Welch et Hedges $g$. Les comparaisons au modèle de référence sont corrigées par la procédure de Holm.

Pour des checkpoints évalués sur les mêmes exemples :

```bash
python experiments/compare_predictions.py \
  --reference fmri=runs/fmri/seed_42/test/predictions.json \
  --comparison gaussian=runs/gaussian/seed_42/test/predictions.json \
  --comparison random=runs/random/seed_42/test/predictions.json \
  --output reports/mcnemar.json
```

## 7. Complexité, robustesse et attention

```bash
python experiments/profile_model.py \
  --checkpoint runs/fer7/fmri/seed_42/best_checkpoint.pt \
  --batch-size 1 --warmup 20 --repeats 100 \
  --device cuda --output reports/profile.json

python experiments/evaluate_robustness.py \
  --checkpoint runs/fer7/fmri/seed_42/best_checkpoint.pt \
  --output reports/robustness.json

python experiments/evaluate_attention.py \
  --checkpoint runs/fer7/fmri/seed_42/best_checkpoint.pt \
  --stage 4 --max-images 100 \
  --output-dir reports/attention
```

`evaluate_attention.py` accepte en option un CSV `identifier,map_path` pour calculer AUC, NSS, corrélation, divergence KL et similarité avec des cartes oculométriques ou masques indépendants. Sans références, il exporte uniquement les cartes et superpositions; il ne prétend pas valider leur alignement humain.

## 8. Tableaux LaTeX

```bash
python experiments/export_latex_tables.py \
  --statistics-json reports/statistics/statistics.json \
  --profile-json reports/profile.json \
  --robustness-json reports/robustness.json \
  --output reports/generated_results.tex
```

Le fichier généré porte un avertissement indiquant que les valeurs proviennent des sorties expérimentales. Les comparaisons SOTA restent indicatives lorsque les protocoles diffèrent.

## 9. Tests logiciels

```bash
python -m unittest discover -s tests -v
```

Les tests utilisent exclusivement des tableaux synthétiques pour valider le logiciel. Ils ne mesurent aucune performance FER.

## Limites et éléments à fournir

Le code est prêt à exécuter, mais une reproduction scientifique nécessite encore les données sous licence, la carte fMRI de groupe ou les cartes autorisées, le fichier réel de correspondances TPS, ainsi que les détails d’approbation éthique applicables aux données humaines. Les scores historiques du manuscrit ne sont pas intégrés au code : ils doivent être régénérés et vérifiés à partir des runs archivés.

## Licence et citation

Consultez `LICENSE` et `CITATION.cff`. La citation d’un article non encore publié doit conserver son statut de manuscrit soumis ou de prépublication, sans l’annoncer comme article déjà accepté.
