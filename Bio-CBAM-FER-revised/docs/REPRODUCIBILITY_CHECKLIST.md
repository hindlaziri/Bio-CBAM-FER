# Liste de contrôle avant mise à jour du manuscrit

| Vérification | Preuve attendue | Statut initial |
|---|---|---|
| Provenance du prior fMRI | Carte source, checksum, script et métadonnées | À fournir |
| Correspondances TPS | CSV versionné et justification expérimentale | À fournir |
| Approbation éthique | Comité, référence, consentement et conditions de partage | À fournir |
| FER-2013 sept classes | Cinq dossiers de runs complets avec mêmes seeds | À exécuter |
| FER-2013 quatre classes | Noms exacts des quatre classes et runs séparés | À exécuter |
| CK+ | Manifests subject-independent pour chaque fold | À créer/exécuter |
| JAFFE | Manifests subject-independent pour chaque fold | À créer/exécuter |
| Filtrage SSIM | Audit JSON et résultats avec/sans filtrage | À exécuter |
| Ablations d’architecture et de priors | ResNet, CBAM, fMRI, random-matched, Gaussian et generic saliency | À exécuter |
| Statistiques | Moyenne, SD, IC95%, p brut, p Holm, taille d’effet | À générer |
| McNemar | Fichiers de prédictions alignés sur les mêmes exemples | À générer |
| Complexité | Paramètres, FLOPs, latence, FPS, mémoire, matériel | À mesurer |
| Robustesse | Bruit, flou, luminosité, occlusion et translation | À mesurer |
| Interprétabilité | Cartes exportées et, si disponible, référence indépendante | À générer |
| Tableaux LaTeX | Générés automatiquement depuis les JSON | À générer |

## Règle de décision

Une case ne peut passer à « terminé » que si la preuve correspondante est archivée. Les scores historiques du manuscrit ne doivent pas être copiés manuellement dans les sorties du dépôt. Si une preuve n’est pas disponible, le manuscrit doit présenter l’élément comme une limitation ou retirer l’affirmation.
