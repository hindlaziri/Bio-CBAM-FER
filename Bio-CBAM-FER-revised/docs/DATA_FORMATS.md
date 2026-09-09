# Formats de données et artefacts

## FER-2013

Le fichier CSV doit contenir les colonnes `emotion`, `pixels` et `Usage`. Les valeurs de `Usage` reconnues sont exactement `Training`, `PublicTest` et `PrivateTest`. Les labels numériques suivent le codage officiel :

| ID | Classe |
|---:|---|
| 0 | angry |
| 1 | disgust |
| 2 | fear |
| 3 | happy |
| 4 | sad |
| 5 | surprise |
| 6 | neutral |

Le protocole quatre classes sélectionne explicitement quatre classes originales. Il ne fusionne pas de labels. L’étiquette « Confusion » est refusée car elle n’appartient pas à FER-2013.

## Manifests CK+ et JAFFE

Un manifest final doit contenir :

| Colonne | Contenu |
|---|---|
| `path` | Chemin absolu ou relatif au manifest |
| `label` | Nom canonique de l’expression |
| `subject_id` | Identifiant stable du sujet |
| `split` | `train`, `val` ou `test` |
| `id` | Identifiant optionnel de l’image |

Le loader vérifie qu’un même `subject_id` n’apparaît jamais dans plusieurs partitions.

## Cartes d’activation

`fmri_pipeline.py` accepte un tableau 2D/3D `.npy`, une archive `.npz` contenant un seul tableau, une image scalaire ou un volume NIfTI. Pour les volumes 4D, `--volume-index` est obligatoire. Une projection corticale 2D produite par le pipeline de neuro-imagerie reste préférable à une réduction volumique générique.

## Correspondances TPS

Le CSV doit contenir `source_x`, `source_y`, `target_x` et `target_y`. Les coordonnées sont en pixels et suivent l’ordre `(x,y)`. Un minimum de trois points non colinéaires est requis. Les points doivent identifier des correspondances expérimentales documentées; le programme ne les infère pas.

## Priors

Chaque prior `.npy` est un tableau `float32` de forme `[H,W]`, fini et normalisé dans `[0,1]`. Le pipeline fMRI produit également un aperçu `.png` et un `.json` contenant les checksums SHA-256. Une banque de $K$ priors est assemblée par `utils.runtime.load_prior_bank` en `[K,H,W]`.

## Checkpoints

Les checkpoints de format 2 contiennent l’état du modèle, les états optimizer/scheduler/scaler, la configuration, les classes, l’environnement, l’historique et la banque de priors. Ils sont autonomes pour reconstruire l’architecture, mais l’évaluation requiert encore le jeu de données sous licence.

## Sorties expérimentales

| Fichier | Contenu |
|---|---|
| `configuration.json` | Arguments, architecture et environnement |
| `history.json` | Métriques par époque et trajectoire des portes $\lambda_b$ |
| `best_checkpoint.pt` | Checkpoint sélectionné exclusivement sur la validation |
| `last_checkpoint.pt` | Dernier état entraîné, utilisable pour reprendre |
| `test/metrics.json` | Métriques finales du test |
| `test/predictions.json` | Cibles, prédictions, probabilités et identifiants |
| `run_summary.json` | Résumé du run et emplacement du checkpoint sélectionné |

Les fichiers JSON calculés constituent la seule source autorisée pour générer les tableaux numériques du manuscrit.
