# Guide d’utilisation

Le `README.md` principal présente le workflow complet. Ce document sert d’index opérationnel et rappelle les garanties de chaque commande.

| Étape | Commande | Sortie principale |
|---|---|---|
| Prior fMRI/TPS | `python -m priors.generate_fmri_priors --help` | `.npy`, `.png`, métadonnées et checksums |
| Priors de contrôle | `python -m priors.variants --help` | Random-matched, Gaussian ou generic-saliency |
| Manifests | `python -m dataset_scripts.prepare_manifests --help` | CSV avec `subject_id` et folds |
| Entraînement | `python train.py --help` | Checkpoints autonomes et historique |
| Évaluation | `python eval.py --help` | Métriques et prédictions du split demandé |
| Ablations | `python experiments/run_ablation.py --help` | Runs appariés par seed |
| Statistiques | `python experiments/analyze_runs.py --help` | IC95%, tests, effets et correction de Holm |
| McNemar | `python experiments/compare_predictions.py --help` | Comparaisons appariées par exemple |
| Complexité | `python experiments/profile_model.py --help` | Paramètres, FLOPs, latence, FPS et mémoire |
| Robustesse | `python experiments/evaluate_robustness.py --help` | Métriques sous perturbations contrôlées |
| Attention | `python experiments/evaluate_attention.py --help` | Cartes, overlays et métriques optionnelles |
| Tables | `python experiments/export_latex_tables.py --help` | Tables LaTeX générées depuis les JSON |
| Tests | `python -m unittest discover -s tests -v` | Validation logicielle uniquement |

## Principes d’utilisation

L’entraînement sélectionne le meilleur checkpoint sur la validation et n’évalue le test qu’après cette sélection. Un run qui exige le prior doit employer `--require-prior`; ainsi, un chemin oublié provoque une erreur au lieu de lancer silencieusement CBAM sans prior.

FER-2013 suit les partitions officielles. CK+ et JAFFE exigent des manifests séparés par sujet. Le filtrage SSIM, lorsqu’il est activé, modifie uniquement l’entraînement et archive toutes les suppressions.

Les scripts d’analyse ne créent pas de métriques manquantes. Ils échouent si les runs, configurations ou prédictions nécessaires ne sont pas présents. Les valeurs destinées au manuscrit doivent provenir de `export_latex_tables.py`, jamais d’une saisie manuelle dans le dépôt.

## Reprise d’un entraînement

```bash
python train.py \
  --dataset fer2013 --data-path /PATH/fer2013.csv \
  --output-dir runs/fer7/fmri/seed_42 \
  --num-classes 7 --prior /PATH/fmri.npy --require-prior \
  --resume runs/fer7/fmri/seed_42/last_checkpoint.pt
```

La configuration de reprise doit rester cohérente avec le checkpoint. En particulier, la banque de priors, l’ordre des classes et l’architecture ne doivent pas changer.

## Évaluation d’un checkpoint portable

```bash
python eval.py \
  --checkpoint runs/fer7/fmri/seed_42/best_checkpoint.pt \
  --data-path /PATH/fer2013.csv \
  --output-dir reports/fer7_seed42_test
```

L’architecture et la banque de priors sont reconstruites depuis le checkpoint. Le chemin des données peut être fourni à nouveau lorsque la machine change.
