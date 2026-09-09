# Construction des priors spatiaux fMRI/TPS

Ce dossier contient le pipeline de construction des priors utilisés par Bio-CBAM. Il **ne contient pas de données humaines** et ne fabrique pas de cartes dites « fMRI-derived » à partir de poids faciaux codés en dur.

## Entrées obligatoires

Le pipeline nécessite une carte statistique issue de l’analyse fMRI et un fichier CSV de correspondances. La carte peut être une projection corticale 2D exportée par le logiciel de neuro-imagerie, un tableau NumPy, une image scalaire ou un volume NIfTI. Pour un volume 3D, les projections intégrées (`max_abs`, `mean_abs`, `maximum` ou `slice`) servent uniquement à des analyses documentées de sensibilité; elles ne remplacent pas une projection corticale validée.

Le CSV doit contenir les colonnes suivantes :

| Colonne | Définition |
|---|---|
| `source_x`, `source_y` | Coordonnées du repère dans la carte source 2D |
| `target_x`, `target_y` | Coordonnées homologues dans le canevas facial cible |

Ces correspondances sont une **hypothèse expérimentale explicite**. Le programme ne prétend pas qu’elles représentent une projection anatomique du cerveau vers les muscles du visage.

## Commande

Depuis la racine du dépôt :

```bash
python -m priors.generate_fmri_priors \
  --source /chemin/vers/stat_map.npy \
  --correspondences /chemin/vers/correspondences.csv \
  --output-stem artifacts/priors/prior_happy \
  --height 224 --width 224 \
  --activation-mode positive \
  --threshold-percentile 95 \
  --smooth-sigma 2.0 \
  --tps-regularization 0.001
```

Chaque exécution produit un fichier `.npy`, un aperçu `.png` et un fichier `.json` contenant la configuration, les dimensions et les checksums SHA-256 des entrées et de la sortie.

## Utilisation dans le modèle

Une banque de priors de forme `[K, H, W]` peut être transmise à `BioCBAM`. Lorsque `K > 1`, le modèle apprend des poids de mélange à partir des caractéristiques visuelles; il ne sélectionne jamais un prior à partir de l’étiquette vraie, ce qui éviterait une fuite de cible.

```python
import numpy as np
import torch
from models import BioCBAM, BioCBAMConfig

bank = torch.from_numpy(np.stack([
    np.load("artifacts/priors/prior_angry.npy"),
    np.load("artifacts/priors/prior_happy.npy"),
]))
model = BioCBAM(BioCBAMConfig(num_classes=7), prior_bank=bank)
logits, diagnostics = model(images)
```

## Traçabilité et éthique

Les cartes individuelles, métadonnées d’acquisition et identifiants des participants ne doivent pas être publiés sans autorisation. Le dépôt destiné à la reproduction peut contenir les cartes de groupe dérivées et anonymisées, les correspondances, les scripts et les métadonnées, sous réserve de l’approbation du comité d’éthique et des conditions de consentement.
