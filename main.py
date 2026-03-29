## Intégration du pipeline complet

"""
Choix du dataset à entrer : bad_data ou Donnees/ch/Fichiers audio cleaned

Data Cleaning (appeler cleaning_main.py)


Faire une comparaison entre deux modèles de reconnaissance de phonème pour obtenir un intervalle de confiance plus élévé (ET Logique) (ou 3) :
- models/spectrogram_phoneme_classifier.py -> Reseau de neurones ?
- Classification binaire (run_pipeline.py)
- Wave2Vec (tous les 3 Wave2Vec entrainés)

Déterminer l'erreur grace au csv et donc -> soit déterminer le meilleur modèle, soit utiliser les 3 pour être sûr mais surement 'overkill'
"""
