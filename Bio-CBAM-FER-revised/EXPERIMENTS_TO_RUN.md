# Expériences à exécuter avec les données réelles

Le code est prêt, mais les résultats du manuscrit ne sont pas reproductibles tant que les données et artefacts suivants ne sont pas fournis.

## Entrées indispensables

| Élément | Action requise |
|---|---|
| FER-2013 | Fournir le CSV original avec les trois valeurs officielles de `Usage` |
| CK+ | Fournir les images et labels d’expression, puis générer les manifests par sujet |
| JAFFE | Fournir les images sous licence, puis générer les manifests par sujet |
| fMRI | Fournir la carte statistique de groupe ou les cartes autorisées |
| TPS | Remplir `priors/correspondences_TEMPLATE.csv` avec des correspondances réellement justifiées |
| Éthique | Fournir comité, numéro d’approbation, consentement et restrictions de partage |
| Eye-tracking | Facultatif; fournir un manifest `identifier,map_path` pour la validation quantitative de l’attention |

## Ordre d’exécution

1. Construire le prior fMRI/TPS et archiver son JSON de provenance.
2. Construire les priors random-matched pour chaque seed, le Gaussian center et la salience générique.
3. Générer et auditer les manifests CK+/JAFFE.
4. Lancer séparément les protocoles FER-2013 sept classes et quatre classes.
5. Lancer les six variantes d’ablation : ResNet, CBAM, fMRI, random-matched, Gaussian et generic saliency.
6. Lancer la comparaison FER-2013 avec et sans filtrage SSIM, sans modifier les partitions officielles.
7. Agréger les runs et exécuter les tests statistiques.
8. Exécuter McNemar sur des prédictions alignées par identifiant.
9. Mesurer la complexité sur le même matériel pour toutes les architectures.
10. Exécuter robustesse et interprétabilité.
11. Générer automatiquement les tableaux LaTeX.
12. Modifier le manuscrit uniquement après vérification des JSON et checkpoints archivés.

## Conditions minimales pour conserver les scores historiques

Les valeurs historiques 94.7%, 74.8%, 96.3% et 93.7% ne doivent être conservées que si elles sont retrouvées dans des runs complets associés à des checkpoints, configurations, splits et prédictions. Si les nouvelles valeurs diffèrent, le manuscrit et la réponse aux reviewers doivent employer les nouvelles mesures réelles.
