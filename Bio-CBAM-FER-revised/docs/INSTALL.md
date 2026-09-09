# Installation

## Environnement

Le code cible Python 3.10–3.12. Un GPU CUDA est recommandé pour ResNet-50, mais les tests logiciels fonctionnent sur CPU. Les versions exactes de CUDA et de PyTorch doivent être enregistrées avec chaque run; `train.py` le fait automatiquement dans `configuration.json`.

## Installation standard

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour CUDA, installez d’abord le couple PyTorch/torchvision recommandé pour votre pilote depuis [pytorch.org](https://pytorch.org/get-started/locally/), puis installez les autres dépendances. Ne changez pas de version de PyTorch entre les variantes d’une même expérience.

## Vérification

Depuis la racine du projet :

```bash
python -m unittest discover -s tests -v
python train.py --help
python eval.py --help
python -m priors.generate_fmri_priors --help
python experiments/analyze_runs.py --help
```

Une suite de tests réussie valide les composants logiciels dans l’environnement courant. Elle ne valide pas les scores du manuscrit.

## Données

FER-2013 doit être fourni sous forme de CSV avec les partitions officielles. CK+ et JAFFE nécessitent des manifests par sujet; consultez `DATA_FORMATS.md`. Les licences de ces jeux de données peuvent interdire leur redistribution, et aucune image n’est incluse dans ce package.

Le prior fMRI nécessite une carte statistique autorisée et un fichier réel de correspondances TPS. Le package ne fournit pas de données de participants.

## Dépendances optionnelles intégrées

`nibabel` permet de lire NIfTI. `thop` calcule les FLOPs dans le script de profilage. Si `thop` n’est pas installé, le champ correspondant reste `null` plutôt que d’être estimé ou inventé.

## Dépannage

| Problème | Vérification recommandée |
|---|---|
| Mémoire CUDA insuffisante | Réduire `--batch-size`, activer `--amp`, ou utiliser ResNet-18 pour les tests |
| Split FER vide | Vérifier les valeurs exactes `Training`, `PublicTest`, `PrivateTest` |
| Fuite de sujets | Corriger le manifest; le loader refuse automatiquement le chevauchement |
| Checkpoint incompatible | Utiliser exclusivement les checkpoints `format_version=2` créés par cette version |
| Prior absent | Fournir `--prior ... --require-prior`; sinon le run est explicitement marqué `prior_mode=none` |
| NIfTI illisible | Vérifier `nibabel`, les dimensions et `--volume-index` pour une entrée 4D |
