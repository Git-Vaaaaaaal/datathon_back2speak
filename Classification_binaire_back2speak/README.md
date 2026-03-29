# Back2speaK — Classification Binaire du Phonème 'che'

Détection automatique des troubles de prononciation du phonème 'che' chez l'enfant.
**Tâche** : classification binaire — Correcte / Incorrecte.

---

## Architecture du projet

```
back2speak/
├── data/
│   ├── processed/      features.npz, splits.npz
│   └── splits/
├── models/
│   ├── architectures.py   CNN PyTorch (optionnel)
│   └── checkpoints/       modèles .joblib sauvegardés
├── src/
│   ├── utils.py           chemins, seed, logger
│   ├── data_loader.py     chargement CSV + labels binaires
│   ├── preprocessing.py   load audio, normalisation, augmentation
│   ├── features.py        extraction MFCC, spectral, chroma…
│   ├── train.py           pipelines sklearn + cross-validation
│   └── evaluate.py        métriques, figures, rapport
├── notebooks/
│   └── exploration.ipynb  EDA + pipeline interactif complet
├── results/               figures et CSV sauvegardés
├── run_pipeline.py        script principal (CLI)
└── requirements.txt
```

---

## Installation

```bash
# Depuis le répertoire Datathon/back2speak/
python -m pip install -r requirements.txt
```

---

## Utilisation

### Option 1 — Notebook (recommandé pour la présentation)
```bash
cd notebooks/
jupyter notebook exploration.ipynb
```

### Option 2 — Script CLI complet
```bash
# Depuis Datathon/back2speak/
python run_pipeline.py              # pipeline complet
python run_pipeline.py --skip-features  # si features déjà extraites
python run_pipeline.py --model svm_rbf  # un seul modèle
```

### Option 3 — Modules individuels
```bash
# Extraction de features
python -m src.features

# Entraînement + CV
python -m src.train --model all --save --cv 5

# Évaluation
python -m src.evaluate --model models/checkpoints/svm_rbf.joblib
```

---

## Approche technique

### Pourquoi des features classiques + SVM/RF ?

| Critère | Justification |
|---------|--------------|
| **Petit dataset** | 368 samples → deep learning risque l'overfitting |
| **Déséquilibre sévère** | ~94% correct → SMOTE + `class_weight='balanced'` |
| **Audio court** | Phonèmes de 1-3 s → features statistiques MFCC suffisantes |
| **Interprétabilité** | Cliniquement utile de comprendre les features discriminantes |

### Pipeline de features (~205 dimensions)

| Feature | Dim | Description |
|---------|-----|-------------|
| MFCC (mean+std) | 80 | 40 coefficients cepstraux × mean/std |
| Δ-MFCC (mean+std) | 80 | Dérivée temporelle des MFCC |
| Spectral centroid | 2 | Centre de gravité spectral |
| Spectral bandwidth | 2 | Largeur de bande spectrale |
| Spectral rolloff | 2 | Fréquence de rolloff (85%) |
| Zero-crossing rate | 2 | Taux de passage par zéro |
| RMS energy | 2 | Énergie RMS |
| Chroma STFT | 24 | 12 classes de hauteur × mean/std |
| Spectral contrast | 7 | Contraste par bande |
| Métadonnées | 4 | âge, sexe, position, type_item |

### Modèles comparés (CV 5-fold stratifié)
- Logistic Regression (baseline)
- **SVM RBF** (généralement meilleur sur petits datasets)
- Random Forest
- Gradient Boosting

### Gestion du déséquilibre
- **SMOTE** dans le pipeline (appliqué uniquement sur train, jamais test)
- `class_weight='balanced'` dans tous les classifieurs

---

## Métriques suivies

- **F1-score** (métrique principale — équilibre précision/rappel)
- **AUC-ROC** (robuste au déséquilibre)
- Accuracy, Precision, Recall
- Matrice de confusion

> Note clinique : le **rappel** (recall) sur la classe "Incorrecte" est critique —
> un faux négatif (prononcer incorrectement sans le détecter) est plus coûteux
> qu'un faux positif.

---

## Résultats (référence)

| Modèle | F1 (CV) | AUC (CV) |
|--------|---------|---------|
| SVM RBF | ~0.70 | ~0.85 |
| Random Forest | ~0.65 | ~0.82 |
| Gradient Boosting | ~0.68 | ~0.83 |
| Logistic Regression | ~0.60 | ~0.78 |

*Résultats indicatifs — varient selon le split*

---

## Pistes d'amélioration

1. **Transfer learning** : Wav2Vec2/HuBERT pré-entraîné sur speech (besoin de plus de GPU)
2. **Leave-One-Speaker-Out CV** : évaluation plus réaliste (7 locuteurs)
3. **Collecte de données** : plus d'exemples "incorrects" (classe minoritaire)
4. **Threshold tuning** : abaisser le seuil de décision pour maximiser le rappel
