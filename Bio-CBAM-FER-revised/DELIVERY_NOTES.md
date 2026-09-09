# Livraison du code Bio-CBAM révisé

## Ce qui a été préparé

Le dépôt original a été conservé séparément. La version révisée fournit maintenant une implémentation cohérente et testée de l’architecture décrite : ResNet multi-échelle, CBAM après quatre stages et fusion additive du prior selon $M_b=\sigma(Z_b+\lambda_bH_b)$. Les portes $\lambda_b$ sont apprenables, configurables, archivées dans les checkpoints et suivies pendant l’entraînement.

Le pipeline de prior charge une carte réelle, documente la réduction 3D–2D, applique le seuil et le lissage choisis, exige des correspondances TPS explicites, puis écrit la carte et ses métadonnées. Aucune carte fMRI ni correspondance anatomique n’est fabriquée.

Les loaders respectent les partitions FER-2013 et refusent « Confusion ». CK+ et JAFFE utilisent des manifests avec `subject_id`; toute fuite de sujet provoque une erreur. Le SSIM ne filtre que le training set et génère un audit.

Les expériences couvrent ResNet, CBAM, Bio-CBAM/fMRI, random-matched propre à chaque seed, Gaussian center et generic saliency. Les sorties permettent IC95%, tests appariés ou Welch, tailles d’effet, correction de Holm et McNemar exact. Sont également inclus le profilage de complexité, les perturbations de robustesse, l’export d’attention et la génération automatique de tableaux LaTeX.

## Validation effectuée

L’analyse statique ne signale aucune erreur. Les 12 tests unitaires réussissent. Un workflow synthétique temporaire de bout en bout réussit : entraînement, checkpoint autonome, évaluation, profilage, robustesse, attention et export LaTeX. Les artefacts synthétiques sont supprimés automatiquement et ne constituent pas des résultats scientifiques.

## Ce qui reste à exécuter

Les datasets, cartes fMRI, correspondances TPS, informations d’éthique, checkpoints et logs réels ne faisaient pas partie des fichiers fournis. Les performances historiques du manuscrit ne sont donc pas reproduites ni codées en dur. Elles devront être recalculées avec les données autorisées avant la réponse finale aux reviewers.
