# Rapport de validation du code révisé

## Résultat

Le package a été validé le **2 septembre 2026** dans un environnement CPU Python 3.12.3. Cette validation porte sur le logiciel; elle ne constitue pas une reproduction des scores du manuscrit.

| Contrôle | Résultat |
|---|---|
| Compilation de tous les modules Python | Réussie |
| Analyse statique Ruff | Réussie, aucune alerte |
| Tests unitaires | 12/12 réussis |
| Smoke test entraînement d’une époque | Réussi sur données synthétiques temporaires |
| Reconstruction du checkpoint autonome | Réussie |
| Évaluation depuis checkpoint | Réussie |
| Profilage paramètres/FLOPs/latence | Réussi |
| Robustesse sur deux conditions de smoke test | Réussie |
| Export d’une carte d’attention | Réussi |
| Génération de tableaux LaTeX depuis JSON | Réussie |
| Dry-run des ablations multi-seeds | Réussi pour ResNet, CBAM et quatre priors Bio-CBAM |

## Couverture fonctionnelle

| Remarque méthodologique des reviewers | Réponse apportée dans le code |
|---|---|
| Mapping fMRI–visage non défini | Pipeline TPS explicite exigeant une carte et des correspondances versionnées; aucune correspondance anatomique n’est inventée |
| Paramètre $\lambda$ insuffisamment défini | Une porte apprenable par stage, initialisation configurable, partage optionnel, trajectoire et valeur finale enregistrées |
| Sélection du prior pouvant fuir l’étiquette | Le mélange d’une banque de priors dépend exclusivement des caractéristiques visuelles, jamais de la cible |
| Classe « Confusion » absente de FER-2013 | Le loader refuse cette étiquette et le protocole quatre classes est un sous-ensemble strict |
| Splits FER-2013 | Utilisation exclusive de `Training`, `PublicTest`, `PrivateTest` |
| CK+/JAFFE non subject-independent | Manifests obligatoires avec `subject_id` et contrôle de disjonction |
| Filtrage SSIM pouvant modifier le test | Filtrage limité au training, audit complet; validation/test inchangés |
| Baselines insuffisantes | Trois architectures séparées : ResNet, CBAM et Bio-CBAM |
| Priors de contrôle | Random-matched propre à chaque seed, Gaussian center et generic spectral saliency |
| Statistiques incomplètes | IC95%, test apparié ou Welch, Cohen $d_z$/Hedges $g$, correction de Holm et McNemar exact |
| Coût et robustesse | Profilage FLOPs/FPS/mémoire/convergence et perturbations paramétrées |
| Interprétabilité qualitative seulement | Export numérique des cartes et métriques optionnelles avec références indépendantes |
| Faux poids ou faux runs | Générateur de checkpoints factices supprimé; aucune métrique historique codée en dur |

## Limite de cette validation

Aucune donnée FER-2013, CK+, JAFFE, fMRI ou eye-tracking n’était incluse. Par conséquent, aucun score réel, test statistique, tableau de comparaison ou conclusion biologique n’a été généré. Ces éléments doivent être obtenus en exécutant le protocole documenté sur les données autorisées.
