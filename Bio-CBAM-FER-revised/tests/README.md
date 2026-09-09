# Tests logiciels

Les tableaux synthétiques créés par `test_core.py` servent uniquement à vérifier les dimensions, les erreurs attendues, les invariants du TPS, les splits, le filtrage SSIM et les calculs statistiques. **Ils ne doivent jamais être cités comme résultats du modèle.**

Exécution depuis la racine du dépôt :

```bash
python -m unittest discover -s tests -v
python tests/smoke_training.py
```

Le second test exécute temporairement un entraînement d’une époque, puis l’évaluation, le profilage, deux conditions de robustesse, l’export d’attention et la génération de tableaux LaTeX. Tous les artefacts synthétiques sont supprimés à la fin.

Une exécution réussie établit que les composants logiciels testés fonctionnent dans l’environnement courant; elle ne valide ni la provenance des cartes fMRI, ni la pertinence biologique des correspondances TPS, ni les scores de reconnaissance sur des données réelles.
