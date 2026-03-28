# Dossier `src/` — Les modules Python

Ce dossier contient le code Python qui fait le lien entre
les résultats du modèle d'analyse phonétique et l'ontologie OWL.

---

## `ontology_populator.py` — Le module principal

### À quoi ça sert ?

Ce module est le cœur du système. Il s'occupe de :

- **Charger** l'ontologie OWL depuis le fichier `ontologie.owx`
- **Créer automatiquement** les individus (patients, exercices, phonèmes)
- **Éviter les doublons** grâce à un système de mémoire interne
- **Sauvegarder** l'ontologie enrichie dans un fichier .owl

### Ce qu'il contient

#### `PatientInfo` — Fiche patient

Un simple conteneur pour les informations d'un patient :

```
patient_id    → identifiant unique, ex: "P02_S02"
speaker_id    → identifiant du locuteur, ex: "S02"
age           → âge du patient
sexe          → "F" ou "M"
collecteur_id → orthophoniste responsable, ex: "P02"
```

#### `OntologyPopulator` — Le peupleur d'ontologie

La classe principale. Elle contient toutes les fonctions pour
ajouter des données dans l'ontologie.

**Fonctions importantes :**

| Fonction | Ce qu'elle fait |
|----------|----------------|
| `__init__(chemin_ontologie)` | Charge l'ontologie au démarrage |
| `add_error_from_model_output(patient, erreur)` | Ajoute une erreur dans l'ontologie |
| `parse_model_output(texte)` | Transforme la sortie du modèle en liste d'erreurs |
| `get_statistics()` | Compte les patients, exercices, phonèmes |
| `print_statistics()` | Affiche les statistiques dans la console |
| `save_ontology(chemin_sortie)` | Sauvegarde le fichier .owl |

**Ce qui se passe quand on ajoute une erreur :**

Imaginons que l'enfant dise "flèse" au lieu de "flèche" :

```
1. On identifie ou crée le patient (P02_S02)
2. On crée le phonème travaillé : /ʃ/ en position finale (Fin)
   → niveau de difficulté = 2
3. On note le phonème précédent : /ɛ/ (le son "è" avant le "ch")
4. On crée une catégorie de mots pour ce phonème à cette position
5. On crée un exercice en cours pour ce patient
   → niveau = 2, taux de réussite initial = 0
6. On lie le patient à son exercice
```

**Règle des niveaux (très important pour les orthophonistes) :**

```
Position DEBUT de mot  →  Niveau 3  (le plus difficile)
Position FIN de mot    →  Niveau 2  (difficile)
Position MILIEU de mot →  Niveau 1  (le moins difficile)
```

---

## `data_processor.py` — Le module de données

### À quoi ça sert ?

Ce module lit les fichiers CSV du projet et prépare les données
pour être ajoutées dans l'ontologie.

### Ce qu'il contient

#### `ItemMetadata` — Métadonnées d'un item phonétique

Contient toutes les infos sur un item (un mot ou une syllabe du catalogue) :
position du phonème cible, voyelle qui précède, voyelle qui suit, etc.

#### Fonctions disponibles

**`load_patients_from_csv(chemin_csv)`**

Lit le fichier `audio_db.csv` et retourne un dictionnaire de patients.
Un même patient peut apparaître plusieurs fois dans le CSV (une ligne par
enregistrement audio) — cette fonction ne garde qu'une entrée par patient.

**`load_items_metadata(chemin_csv)`**

Lit le fichier `exercices_a_realiser.csv` et retourne le catalogue
complet des items phonétiques.

**`process_batch_from_csv(populator, csv, sorties_modele)`**

Traite automatiquement plusieurs enregistrements d'un coup :
1. Lit la liste des patients depuis le CSV
2. Pour chaque enregistrement audio, retrouve le patient
3. Analyse la sortie du modèle
4. Ajoute toutes les erreurs dans l'ontologie

**`build_errors_from_items(items, phoneme, filtre_type)`**

Construit une liste d'erreurs à partir du catalogue d'items.
Utile pour peupler l'ontologie avec tous les scénarios possibles
sans avoir besoin des vraies sorties du modèle.

---

## Comment les deux modules interagissent

```
audio_db.csv                exercices_a_realiser.csv
      │                              │
      ▼                              ▼
data_processor.py          data_processor.py
load_patients_from_csv()   load_items_metadata()
      │                              │
      │    PatientInfo               │    ItemMetadata
      └──────────────┐    ┌──────────┘
                     ▼    ▼
              ontology_populator.py
              OntologyPopulator.add_error_from_model_output()
                          │
                          ▼
                   ontologie_enrichie.owl
```

---

## Pour les non-développeurs : analogie simple

Imaginez une **bibliothécaire** (OntologyPopulator) qui gère un grand
fichier de suivi orthophonique (l'ontologie OWL).

Quand un enregistrement audio est analysé par le modèle IA, on donne
à la bibliothécaire une fiche avec :
- Qui est le patient ?
- Quel son a-t-il raté ?
- À quelle position dans le mot ?
- Qu'y avait-il avant et après ce son ?

La bibliothécaire vérifie si elle connaît déjà ce patient, ce son, cet exercice.
Si oui, elle réutilise ce qui existe déjà (pas de doublon).
Si non, elle crée une nouvelle entrée.

Finalement, elle met à jour le grand fichier de suivi.
