# Datathon 2026 - Back2Speak

## Introduction

Fort de l’expérience passée avec Back2smilE nous souhaitons avec Back2speaK aborder un nouveau champ de compétence par l’analyse sonore. En France, les troubles de l'articulation concernent entre 5 et 8% des enfants d'âge scolaire selon l'Assurance Maladie. Cette prévalence varie selon l'âge : elle atteint 15% chez les enfants de 3 ans, puis diminue progressivement pour se stabiliser autour de 2-3% à l'adolescence. Les cabinets d’orthophonie sont sur liste d’attente il faut en moyenne 8 à 24 mois (ou plus)pour une prise en charge, les séances sont proposées à une fréquence de 1/7 ce qui est peu pour automatiser un geste articulatoire correct. La répétition quotidienne permettrait d’ancrer le geste plus efficacement et plus rapidement. Nous cherchons à dévelloper un outil de rééducation capable de détecter les déficits phonétiques et de proposer aux patients des exercices à domicile, qui s’adaptent aux besoins, prodiguent des conseils et des exercices efficaces et qui évoluent en fonction des progrès. 

## Comment on fait

 ```mermaid
graph LR

    %% Définition des styles (couleurs et bordures)

    classDef source fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000

    classDef prep fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000

    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000

    classDef logic fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000

    classDef goal fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000

  

    %% Section Données

    subgraph Data ["📊 Data & Préparation"]

        direction TB

        A([🎵 Audio]):::source --> B{Format Image}:::prep

        B --> C[Spectrogrammes]:::prep

        B --> D[Transformées de Fourier]:::prep

        E[(📂 Open Data)]:::source --> F[Data Augmentation]:::prep

        F --> F1[Modification fichiers]:::prep

        F --> F2[Génération / Torch Audio]:::prep

    end

  

    %% Section IA

    subgraph IA ["🧠 Intelligence Artificielle"]

        direction TB

        G([⚙️ Features Extraction]):::model --> H[Manipulation des poids]:::model

        H --> I{Classification}:::model

        I --> I1[**ML :** SVM, Random Forest]:::model

        I --> I2[**DL :** CNN, LSTM, GAN]:::model

        I --> I3[**Attention :** Wav2Vec, BERT]:::model

    end

  

    %% Section Ontologie et Validation

    subgraph Logic ["⚖️ Ontologie & Validation"]

        direction TB

        J((Ontologie)):::logic -- "Auto-alimentation" --> J

        K([✅ Validation de la décision]):::logic

    end

  

    %% Section Objectifs

    subgraph Objectives ["🎯 Objectifs / Résultats"]

        direction TB

        L1[📌 Type d'erreur]:::goal

        L2[📈 Évaluation globale]:::goal

        L3[🔢 Nombre d'erreurs]:::goal

        L4[📍 Localisation dans le mot]:::goal

    end

  

    %% Connexions transversales (Le Pipeline)

    C & D & F1 & F2 -->|Données traitées| G

    I1 & I2 & I3 -->|Prédictions| J

    J --> K

    K --> L1

    K --> L2

    K --> L3

    K --> L4
 ```
