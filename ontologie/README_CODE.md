# Back2speaK Ontologie — Documentation Technique

Documentation destinée aux développeurs qui souhaitent comprendre,
maintenir ou étendre le système.

---

## Architecture générale

```
OntologyPopulator          DataProcessor
      │                         │
      │  load()                 │  load_patients_from_csv()
      │  ← ontologie.owx        │  ← audio_db.csv
      │                         │
      │  add_error_from_model_output(PatientInfo, error_dict)
      │      │
      │      ├─ _get_or_create_patient()
      │      ├─ _get_or_create_ph_travail()
      │      ├─ _get_or_create_ph_context()   ← précédent + suivant
      │      ├─ _get_or_create_categorie()
      │      └─ _get_or_create_exercise()
      │
      │  save_ontology() → ontologie_enrichie.owl
```

---

## Dépendances

| Librairie   | Version min | Usage |
|-------------|-------------|-------|
| `owlready2` | 0.40+       | Manipulation de l'ontologie OWL |
| `csv`       | stdlib      | Lecture des fichiers CSV |
| `re`        | stdlib      | Parsing du texte du modèle |

Installation :
```bash
pip install owlready2
```

---

## Module `ontology_populator.py`

### Classe `PatientInfo`

Dataclass simple, sans logique. Stocke les métadonnées d'un patient.

```python
@dataclass
class PatientInfo:
    patient_id:    str   # ex: "P02_S02" (collecteur + speaker)
    speaker_id:    str   # ex: "S02"
    age:           int   # en mois pour les enfants, années pour adultes
    sexe:          str   # "F" ou "M"
    collecteur_id: str   # ex: "P02"
```

### Classe `OntologyPopulator`

**Initialisation :**

```python
populator = OntologyPopulator("ontologie.owx")
```

Charge l'ontologie OWL via `owlready2`. Le fichier `.owx` est au format
OWL/XML (supporté nativement). L'IRI de base est extrait automatiquement.

**Tables de correspondance :**

`_POSITION_MAP` — normalise ~20 variantes de position vers `Debut`, `Fin`, `Milieu`

`_LEVEL_MAP` — mappe les positions normalisées aux niveaux de difficulté (1-3)

`_PHONEME_TYPE_MAP` — mappe les phonèmes IPA aux valeurs de l'ontologie :
- `/ʃ/` → `"Ch"` (propriété `aTypeTravail`)
- `/ʒ/` → `"Ze"` (propriété `aTypeTravail`)

**Système de cache :**

Quatre dictionnaires Python évitent la création de doublons OWL :

```python
self._patients:   Dict[str, object]  # clé: patient_id
self._ph_travail: Dict[str, object]  # clé: "phoneme__position"
self._ph_context: Dict[str, object]  # clé: "ClassName__phoneme"
self._categories: Dict[str, object]  # clé: "phoneme__position"
self._exercises:  Dict[str, object]  # clé: "patient_name__ph_name"
```

**Nommage des individus OWL :**

La méthode `_sanitize()` remplace les symboles IPA par des équivalents ASCII :
- `ʃ` → `SH`, `ʒ` → `ZH`, `ɛ` → `E_ouv`, etc.

La méthode `_unique_name()` ajoute un suffixe numérique si le nom est déjà pris.

**Flux de `add_error_from_model_output()` :**

```
1. _normalize_position(error["position"])
       "final (end of word)" → "Fin"

2. _get_or_create_patient(patient_info)
       Crée Patient("P02_S02") ou retourne le cached

3. _get_or_create_ph_travail(expected, position)
       Crée phTravail("phT_SH_Fin")
       → ph.aPosition = ["Fin"]
       → ph.aTypeTravail = ["Ch"]

4. _get_or_create_ph_context(preceded_by, "phPrecedent")  # si non vide
       Crée phPrecedent("phPrec_E_ouv")
       → ph.aType = ["ɛ"]
       ph_travail.aPrecedent.append(ph_prec)

5. _get_or_create_ph_context(followed_by, "phSuivant")    # si non vide
       → idem pour phSuivant

6. _get_or_create_categorie(expected, position)
       Crée CategorieMot("Cat_SH_Fin")
       categorie.aPhoneme.append(ph_travail)

7. _get_or_create_exercise(patient, ph_travail, categorie, position)
       Crée ExoEnCours("Exo_P02_S02_phT_SH_Fin")
       → exo.aNiveau = [2]        ← Fin = niveau 2
       → exo.aTauxReussite = [0]
       → exo.aCategorieMots = [categorie]

8. patient.pratique.append(exo)   ← liaison finale
```

**Méthode `parse_model_output()` — expressions régulières :**

| Ce qu'on cherche | Regex utilisée |
|-----------------|----------------|
| IPA de référence | `r"Reference IPA.*?:\s*(\S+)"` |
| Blocs d'erreurs | `re.split(r"Error\s+\d+\s+of\s+\d+\s*:", text)` |
| Phonème attendu | `r"Expected phoneme\s*:\s*\[([^\]]+)\]"` |
| Position | `r"Position in word\s*:\s*(.+?)(?:\n|$)"` |
| Phonème précédent | `r"Preceded by\s*:\s*\[([^\]]*)\]"` |

---

## Module `data_processor.py`

### `load_patients_from_csv(csv_path)`

Lit `audio_db.csv`, déduplique par `speaker`, retourne `Dict[str, PatientInfo]`.

Colonnes requises : `audio_id`, `collecteur_id`, `speaker`, `age`, `sexe`

### `load_items_metadata(csv_path)`

Lit `exercices_a_realiser.csv`, retourne `Dict[str, ItemMetadata]`.

Colonnes requises : `item_id`, `mot / stimulus`, `type`, `position_du_/ʃ/`,
`Nombre de phonèmes`, `Forme`, `Voyelle_précédente`, `Classe_voyelle_précédente`,
`voyelle_suivante`, `classe_voyelle_suivante`

### `process_batch_from_csv(populator, label_csv_path, model_outputs, verbose)`

Orchestre le traitement en lot :
1. Charge les patients via `load_patients_from_csv()`
2. Pour chaque `audio_id`, extrait le `speaker_id` avec `_extract_speaker_from_audio_id()`
   - Format attendu : `COLLECTEUR_SPEAKER_ITEM.wav` → `parts[1]`
3. Parse la sortie modèle avec `populator.parse_model_output()`
4. Appelle `populator.add_error_from_model_output()` pour chaque erreur

Retourne un dictionnaire de statistiques :
```python
{
    "files_processed": int,
    "files_skipped":   int,
    "errors_added":    int,
    "errors_skipped":  int,
}
```

---

## Structure de l'ontologie OWL

### Classes et hiérarchie

```
owl:Thing
├── Patient
├── Exercice
│   ├── ExoEnCours
│   └── ExoSuivant
├── Phoneme
│   ├── phTravail
│   ├── phPrecedent
│   └── phSuivant
└── CategorieMot
```

### Propriétés d'objet (ObjectProperty)

| Propriété       | Domaine      | Range        | Description |
|-----------------|-------------|-------------|-------------|
| `pratique`      | Patient      | ExoEnCours   | Patient → exercice en cours |
| `prochainExo`   | ExoEnCours   | ExoSuivant   | Exercice suivant à faire |
| `aCategorieMots`| Exercice     | CategorieMot | Catégorie de mots de l'exercice |
| `aPhoneme`      | CategorieMot | phTravail    | Phonème travaillé dans la catégorie |
| `aPrecedent`    | Phoneme      | Phoneme      | Phonème précédant le phonème cible |
| `aSuivant`      | Phoneme      | Phoneme      | Phonème suivant le phonème cible |

### Propriétés de données (DataProperty)

| Propriété       | Domaine      | Range              | Valeurs |
|-----------------|-------------|---------------------|---------|
| `aNiveau`       | Exercice     | xsd:integer        | 1, 2, 3 |
| `aTauxReussite` | Exercice     | xsd:integer        | 0–100 |
| `aPosition`     | phTravail    | positionPhoneme    | `Debut`, `Fin`, `Milieu` |
| `aTypeTravail`  | phTravail    | typePhTravail      | `Ch`, `Ze` |
| `aType`         | Phoneme      | string             | valeur IPA |
| `peutContinuer` | Patient      | xsd:boolean        | true, false |

### Règles SWRL intégrées

L'ontologie inclut deux règles SWRL :

**Réussite ≥ 80%** → `peutContinuer = true`, niveau du prochain exercice = niveau + 1

**Échec < 80%** → `peutContinuer = false`, niveau du prochain exercice = niveau - 1

Ces règles sont exécutées par un reasoner OWL (Pellet, HermiT) — pas par Python.

---

## Gestion des erreurs

| Situation | Comportement |
|-----------|-------------|
| Fichier ontologie introuvable | `FileNotFoundError` avec message clair |
| `owlready2` non installé | `ImportError` avec instruction `pip install` |
| Position inconnue | Valeur par défaut `"Milieu"` + message console |
| Phonème non mappé (`_PHONEME_TYPE_MAP`) | `aTypeTravail` non renseigné |
| Valeur preceded_by = "(none)" | Phonème précédent ignoré |
| Exception lors de l'ajout batch | Erreur loggée, traitement continue |

---

## Extension du système

### Ajouter un nouveau phonème travaillé (ex: /f/)

Dans `ontology_populator.py`, ajouter dans `_PHONEME_TYPE_MAP` :
```python
"f": "F",   # si une valeur "F" est ajoutée dans typePhTravail
```

Et dans l'ontologie OWL, étendre la `DatatypeDefinition` de `typePhTravail`.

### Ajouter une nouvelle position

Ajouter dans `_POSITION_MAP` :
```python
"interne": "Milieu",
"entrée":  "Debut",
```

### Utiliser un autre format de sortie modèle

Surcharger `parse_model_output()` ou créer une nouvelle méthode :
```python
def parse_my_format(self, data: dict) -> list:
    return [{
        "expected_phoneme": data["target"],
        "produced_phoneme": data["actual"],
        "position": data["word_position"],
        ...
    }]
```

---

## Tests rapides

```python
# Vérifier que l'ontologie se charge
from src.ontology_populator import OntologyPopulator
pop = OntologyPopulator("ontologie.owx")
print(pop.onto.classes())  # Liste les classes

# Vérifier la normalisation des positions
assert pop._normalize_position("initial") == "Debut"
assert pop._normalize_position("final (end of word)") == "Fin"
assert pop._normalize_position("Médiane") == "Milieu"

# Vérifier le mapping des phonèmes
assert pop._map_phoneme_type("ʃ") == "Ch"
assert pop._map_phoneme_type("ʒ") == "Ze"
assert pop._map_phoneme_type("s") is None

print("Tous les tests passent.")
```
