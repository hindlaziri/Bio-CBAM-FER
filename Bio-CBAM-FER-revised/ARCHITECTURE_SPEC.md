# Bio-CBAM — Spécification d’implémentation révisée

## Objectif

Cette version doit fournir une implémentation exécutable et traçable de Bio-CBAM sans coder en dur de résultats scientifiques. Les valeurs publiables sont exclusivement celles produites par les journaux d’expériences sur les données réelles.

## Architecture

Le backbone ResNet expose les sorties des quatre stages résiduels `layer1` à `layer4`. Un bloc Bio-CBAM est appliqué après chaque stage. Pour un tenseur de caractéristiques $F_b$, le CBAM calcule d’abord l’attention de canal, puis une logit d’attention spatiale $Z_b$. Le prior spatial $H_b$ est obtenu en redimensionnant la carte source à la résolution du stage par interpolation bilinéaire. La fusion est définie par

$$M_b = \sigma\left(Z_b + \lambda_b H_b\right), \qquad F'_b = F_b \odot M_b.$$

Chaque stage possède son propre scalaire apprenable $\lambda_b$. Les portes sont initialisées à une valeur configurable, enregistrées dans les checkpoints et exportées dans les rapports. Une option permet de partager une porte unique pour réaliser une ablation, mais le réglage principal utilise une porte par stage.

Trois architectures sont séparées dans le code. `resnet` ne contient aucun module d’attention; `cbam` contient l’attention canal-spatiale standard sans porte $\lambda$ et sans chemin de prior; `biocbam` ajoute la fusion du prior et les portes apprenables. Cette séparation permet une attribution correcte du gain entre backbone, CBAM et prior externe.

## Priors

Le pipeline accepte des cartes 2D déjà dérivées (`.npy`, `.npz`, NIfTI projeté) et, lorsque les correspondances sont fournies, calcule un recalage Thin-Plate Spline entre des points source 2D et des landmarks faciaux cibles. Le code ne prétend pas déduire automatiquement une correspondance anatomique cerveau–visage : le fichier de correspondances est un artefact expérimental explicite et versionné.

Quatre familles de priors doivent être disponibles pour les ablations : `fmri`, `random_matched`, `gaussian_center` et `generic_saliency`. Les priors générés doivent être enregistrés avec leurs métadonnées, leur seed, leur checksum et leurs paramètres.

## Jeux de données

FER-2013 utilise les partitions officielles `Training`, `PublicTest` et `PrivateTest`. Le protocole principal conserve les sept classes. Un protocole secondaire à quatre classes ne peut utiliser qu’une liste explicite de classes originales; aucune fusion implicite de labels n’est autorisée. Le loader refuse la classe non officielle « Confusion ».

CK+ et JAFFE utilisent des manifests CSV contenant au minimum `path`, `label`, `subject_id` et `split` ou, à défaut, un générateur de folds GroupKFold fondé sur `subject_id`. Les identités ne doivent jamais être partagées entre entraînement, validation et test.

Le filtrage SSIM est exécuté uniquement sur le training set. Il produit un manifeste d’audit listant les paires détectées et les échantillons retirés. Les partitions de validation et de test restent inchangées. Le code ne présente pas SSIM comme une garantie d’indépendance des sujets.

## Entraînement et évaluation

Chaque run entraîne réellement un modèle, sélectionne le checkpoint sur la validation et évalue une seule fois le test. Les seeds contrôlent Python, NumPy, PyTorch CPU/GPU et les workers. Les journaux JSON contiennent configuration, versions, matériel, métriques par époque, meilleur checkpoint et métriques finales.

Le script multi-runs agrège les résultats réels, calcule moyenne, écart-type, IC95%, différence appariée lorsque les runs sont appariés, test de Welch lorsque les groupes sont indépendants, taille d’effet et correction de Holm pour les comparaisons multiples.

## Interprétabilité, robustesse et complexité

Les cartes d’attention sont exportées en valeurs numériques et en visualisations. Les métriques IoU/AUC ne sont calculées que si un masque ou une carte de référence est fourni. Les tests de robustesse appliquent des transformations déterministes paramétrables (bruit, flou, occlusion, luminosité, désalignement) et enregistrent la baisse de performance.

L’analyse de complexité rapporte paramètres entraînables, FLOPs lorsque `thop` est installé, latence avec warm-up, FPS, mémoire maximale CUDA, temps par époque et époque de convergence.

## Intégrité scientifique

Les données synthétiques servent uniquement aux tests logiciels. Elles ne doivent jamais alimenter les tableaux de résultats du manuscrit. Les checkpoints factices et le statut « Ready for Publication » sont supprimés. Toute métrique absente reste explicitement marquée comme non calculée jusqu’à exécution sur les données réelles.
