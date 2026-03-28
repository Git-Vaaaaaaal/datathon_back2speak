# Back2speaK — Système d'ontologie phonétique

Ce dossier contient le système qui peuple automatiquement l'ontologie OWL de Back2speaK
à partir des résultats de détection d'erreurs phonétiques.

## Qu'est-ce que ce système fait ?

Lorsque le modèle d'intelligence artificielle détecte une erreur phonétique chez un patient
(par exemple, l'enfant dit "flèse" au lieu de "flèche"), ce système :

1. Identifie le **patient** concerné
2. Crée un **exercice** adapté à son erreur et à sa difficulté
3. Enregistre le **phonème travaillé** (ex : /ʃ/) et sa position dans le mot
4. Note les **sons voisins** (ce qui précède et suit le phonème erroné)
5. Sauvegarde tout dans l'**ontologie OWL** pour le suivi orthophonique

## Structure des fichiers

```
ontologie/
├── ontologie.owx              ← Ontologie OWL de base (à ne pas modifier)
├── src/
│   ├── ontology_populator.py  ← Module principal (peuplement de l'ontologie)
│   └── data_processor.py      ← Lecture des fichiers CSV
├── examples/
│   └── exemple_utilisation.py ← Exemples prêts à lancer
├── output/                    ← Ontologies enrichies générées ici
├── README.md                  ← Ce fichier
└── README_CODE.md             ← Documentation technique détaillée
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- Le module `owlready2`

### Installation de owlready2

```bash
pip install owlready2
```

## Utilisation rapide

### Cas 1 : Ajouter une erreur manuellement

```python
import sys
sys.path.insert(0, "chemin/vers/ontologie/")

from src.ontology_populator import OntologyPopulator, PatientInfo

# Charger l'ontologie
populator = OntologyPopulator("ontologie.owx")

# Définir le patient
patient = PatientInfo(
    patient_id="P02_S02",
    speaker_id="S02",
    age=45,
    sexe="F",
    collecteur_id="P02"
)

# Définir l'erreur détectée
error = {
    "expected_phoneme": "ʃ",        # ce que l'enfant aurait dû dire
    "produced_phoneme": "s",        # ce que l'enfant a dit
    "position":         "final",    # position dans le mot
    "preceded_by":      "ɛ",        # son précédent
    "followed_by":      "(none)",   # aucun son après
}

# Ajouter dans l'ontologie
populator.add_error_from_model_output(patient, error)

# Voir les statistiques
populator.print_statistics()

# Sauvegarder
populator.save_ontology("output/ontologie_enrichie.owl")
```

### Cas 2 : Parser la sortie du modèle directement

```python
# Texte exact produit par le modèle d'analyse
model_text = """
Reference IPA (correct):   flɛʃ
Produced IPA  (detected):  flɛs
Error 1 of 1:
Expected phoneme:   [ʃ]
Produced phoneme:   [s]
Position in word:   final (end of word)
Preceded by:        [ɛ]
Followed by:        [(none)]
"""

errors = populator.parse_model_output(model_text)
for error in errors:
    populator.add_error_from_model_output(patient, error)
```

### Cas 3 : Traitement depuis le CSV

```python
from src.data_processor import process_batch_from_csv

# Dictionnaire {audio_id: sortie_modele}
model_outputs = {
    "P02_S02_W_F01.wav": "... sortie du modèle ...",
    "P02_S02_W_I01.wav": "... sortie du modèle ...",
}

process_batch_from_csv(
    populator=populator,
    label_csv_path="../Donnees/ch/audio_db.csv",
    model_outputs=model_outputs
)
```

## Règles de niveaux de difficulté

Ces règles sont fixées par la pratique orthophonique :

| Position dans le mot | Valeur ontologie | Niveau |
|---------------------|-----------------|--------|
| Début de mot        | `Debut`         | **3** (plus difficile) |
| Fin de mot          | `Fin`           | **2** (difficile) |
| Milieu de mot       | `Milieu`        | **1** (moins difficile) |

Le système normalise automatiquement les valeurs :
`initial`, `initiale`, `début`, `start` → `"Debut"`
`final`, `finale`, `fin`, `end` → `"Fin"`
`medial`, `médiane`, `milieu`, `middle` → `"Milieu"`

## Lancer les exemples

Depuis le dossier `ontologie/` :

```bash
python examples/exemple_utilisation.py
```

## Pour aller plus loin

- **Documentation technique** : voir `README_CODE.md`
- **Description des modules** : voir `src/README.md`
- **Guide des exemples** : voir `examples/README.md`
