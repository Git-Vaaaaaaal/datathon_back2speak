# Datathon 2026 - Back2Speak

## Introduction

Fort de l'expérience passée avec Back2smilE nous souhaitons avec Back2speaK aborder un nouveau champ de compétence par l'analyse sonore. En France, les troubles de l'articulation concernent entre 5 et 8% des enfants d'âge scolaire selon l'Assurance Maladie. Cette prévalence varie selon l'âge : elle atteint 15% chez les enfants de 3 ans, puis diminue progressivement pour se stabiliser autour de 2-3% à l'adolescence. Les cabinets d'orthophonie sont sur liste d'attente il faut en moyenne 8 à 24 mois (ou plus)pour une prise en charge, les séances sont proposées à une fréquence de 1/7 ce qui est peu pour automatiser un geste articulatoire correct. La répétition quotidienne permettrait d'ancrer le geste plus efficacement et plus rapidement. Nous cherchons à développer un outil de rééducation capable de détecter les déficits phonétiques et de proposer aux patients des exercices à domicile, qui s'adaptent aux besoins, prodiguent des conseils et des exercices efficaces et qui évoluent en fonction des progrès.

## Cible phonétique : le son « ch » (ʃ) en français

Le premier phonème ciblé est la consonne fricative palatale **ʃ** (comme dans *chat*, *chien*, *choix*). C'est l'une des erreurs les plus fréquentes chez les enfants présentant des troubles articulatoires : ils la substituent souvent par [s] (sigmatisme) ou l'omettent. La reconnaissance automatique de ce phonème constitue la brique de base du pipeline avant d'étendre à d'autres phonèmes.

**Tâche :** étant donné un enregistrement audio d'un enfant prononçant un mot ou une phrase contenant « ch », le système doit :
1. **Localiser** automatiquement le segment ʃ via Montreal Forced Aligner (`french_mfa`)
2. **Extraire** les features acoustiques pertinentes pour les fricatives (énergie haute fréquence > 4 kHz, centroïde spectral, MFCC, spectrogramme Mel)
3. **Classifier** le phonème : `correct` ou `incorrect` (et à terme : type d'erreur — substitution, distorsion, omission)
4. **Restituer** un feedback adapté au patient

## Structure du code

| Dossier / Fichier | Rôle |
|---|---|
| `main.py` | Orchestrateur du pipeline complet (7 étapes) |
| `Cleaning/cleaning_main.py` | Nettoyage audio par lot : filtrage, normalisation, post-traitement |
| `Cleaning/filtre.py` | Filtre passe-bande et réduction de bruit |
| `Cleaning/normalisateur.py` | Normalisation RMS / peak |
| `pre_processing/mfa_phoneme_extractor.py` | Pipeline MFA complet : aligne audio + transcription, génère TextGrids, extrait les segments ʃ en `.wav` |
| `pre_processing/audio_extractor.py` | Extraction de phonèmes/mots depuis un TextGrid existant (`pydub`, `tgt`) |
| `pre_processing/data_augmentation.py` | Augmentation spectrale : déplacement du centroïde et boost des HF fricatives, génère N variantes par fichier |
| `pre_processing/csv_database_extractor.py` | Mise en forme du CSV de la base de données |
| `Classification_binaire_back2speak/src/features.py` | Extraction de 200+ features acoustiques via `librosa` (MFCC, centroïde, ZCR, RMS, chroma…) |
| `Classification_binaire_back2speak/src/train.py` | Entraînement SVM / Random Forest avec validation croisée |
| `Classification_binaire_back2speak/src/evaluate.py` | Métriques (Accuracy, F1, AUC), courbe ROC, matrice de confusion |
| `Classification_binaire_back2speak/src/data_loader.py` | Chargement et jointure `audio_db_id.csv` ↔ fichiers audio |
| `wav2vec/main.py` | Analyse phonémique fine via `Cnam-LMSSC/wav2vec2-french-phonemizer` : alignement, détection d'erreurs, score de prononciation |
| `ontologie/src/ontology_populator.py` | Peuplement de l'ontologie orthophonique OWL (intégration pipeline à venir) |
| `audio_db_id.csv` | Base annotée : `audio_id`, `speaker`, `age`, `sexe`, `position`, `type_item`, `decision` (correct/incorrect) |

## Pipeline complet (`main.py`)

```mermaid
graph TD

    classDef source  fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef prep    fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef ml      fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef w2v     fill:#e8eaf6,stroke:#3949ab,stroke-width:2px,color:#000
    classDef out     fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#000
    classDef onto    fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000

    RAW([Audio brut\nDonnees/ch/Fichiers audio]):::source

    subgraph Step1 ["[1] Nettoyage — Cleaning/"]
        CLN[cleaning_main.py\nfiltrage · normalisation · post-traitement]:::prep
    end

    subgraph Step2 ["[2] Extraction MFA — pre_processing/"]
        MFA[mfa_phoneme_extractor.py\nalignement forcé french_mfa]:::prep
        SEG[Segments ʃ .wav\nmfa_segments/]:::prep
    end

    subgraph Step3 ["[3] Labels"]
        CSV[(audio_db_id.csv\ncorrect / incorrect)]:::source
        DL[data_loader.py\njointure CSV ↔ audio]:::prep
    end

    subgraph Step4 ["[4] Augmentation — pre_processing/"]
        AUG[data_augmentation.py\ndéplacement centroïde · boost HF >4 kHz\nN variantes par fichier]:::prep
    end

    subgraph Step5 ["[5] Classification ML — Classification_binaire_back2speak/src/"]
        FEAT[features.py\nMFCC · centroïde · ZCR · RMS · chroma · contrast\n~200 features / clip]:::ml
        TRAIN[train.py\nSVM · Random Forest\nvalidation croisée k-fold]:::ml
        EVAL[evaluate.py\nAccuracy · F1 · AUC · ROC · confusion matrix]:::ml
    end

    subgraph Step6 ["[6] Analyse wav2vec — wav2vec/"]
        W2V[main.py\nCnam-LMSSC/wav2vec2-french-phonemizer\nalignement phonémique · détection erreurs\nscore de prononciation 0–100]:::w2v
    end

    subgraph Step7 ["[7] Résultats"]
        PNG1[comparison_metrics.png\nAccuracy · F1 · AUC tous modèles]:::out
        PNG2[comparison_confusion_matrices.png\nML vs wav2vec]:::out
    end

    ONTO["ontologie/\nontology_populator.py\n(intégration à venir)"]:::onto

    RAW --> CLN
    CLN --> MFA
    MFA --> SEG
    SEG & CSV --> DL
    DL --> AUG
    AUG --> FEAT
    FEAT --> TRAIN
    TRAIN --> EVAL
    DL --> W2V
    EVAL --> PNG1
    EVAL --> PNG2
    W2V --> PNG1
    W2V --> PNG2
    EVAL & W2V -.->|prédictions futures| ONTO
```

[Exemple de données public](https://lbourdois.github.io/blog/audio/dataset_audio_fr/)
