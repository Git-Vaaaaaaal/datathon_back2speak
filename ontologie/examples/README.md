# Dossier `examples/` — Guide des exemples

Ce dossier contient des exemples concrets et prêts à lancer
pour utiliser le système d'ontologie Back2speaK.

---

## `exemple_utilisation.py`

Ce fichier regroupe **4 exemples indépendants**, du plus simple au plus complet.

### Comment lancer les exemples

Depuis le dossier `ontologie/` (où se trouve le fichier `ontologie.owx`) :

```bash
python examples/exemple_utilisation.py
```

Par défaut, seul l'**Exemple 1** se lance. Pour en lancer d'autres,
ouvrez le fichier et décommentez les lignes à la fin :

```python
# À la fin du fichier, modifier ces lignes :
exemple_1_ajout_manuel()
# exemple_2_parse_modele()
# exemple_3_traitement_csv()
# exemple_4_items_phonetiques()
```

---

## Description des 4 exemples

### Exemple 1 — Ajout manuel d'une erreur *(le plus simple)*

**Pour qui ?** Tout le monde, pour tester que le système fonctionne.

**Ce qu'il fait :**
- Crée un patient fictif (P02, S02, 45 ans, femme)
- Ajoute l'erreur : le patient dit /flɛs/ au lieu de /flɛʃ/
- Génère l'ontologie enrichie dans `output/ontologie_enrichie.owl`

**Ce qu'on apprend :** comment créer un `PatientInfo` et un dictionnaire d'erreur.

---

### Exemple 2 — Parser la sortie brute du modèle

**Pour qui ?** Développeurs intégrant la sortie du modèle IA.

**Ce qu'il fait :**
- Prend le texte exact produit par le modèle (copié/collé)
- L'analyse automatiquement pour en extraire les erreurs
- Gère plusieurs erreurs dans le même mot

**Ce qu'on apprend :** comment utiliser `parse_model_output()` pour
éviter d'extraire les informations à la main.

---

### Exemple 3 — Traitement par lot depuis le CSV

**Pour qui ?** Traitement automatique de nombreux enregistrements.

**Prérequis :** le fichier `Donnees/ch/audio_db.csv` doit exister.

**Ce qu'il fait :**
- Lit la liste des patients depuis `audio_db.csv`
- Traite plusieurs enregistrements audio d'un coup
- Identifie automatiquement chaque patient depuis le nom du fichier audio

**Ce qu'on apprend :** comment utiliser `process_batch_from_csv()`
pour automatiser le traitement de toute une session.

---

### Exemple 4 — Explorer les items phonétiques

**Pour qui ?** Comprendre le catalogue d'exercices disponibles.

**Prérequis :** le fichier `Donnees/ch/exercices_a_realiser.csv` doit exister.

**Ce qu'il fait :**
- Charge tous les items du catalogue (mots, syllabes, phonèmes isolés)
- Affiche un résumé par type et par position
- Peuple l'ontologie avec tous les scénarios d'exercices possibles

**Ce qu'on apprend :** comment explorer les métadonnées phonétiques
et utiliser `build_errors_from_items()`.

---

## Interpréter les résultats

Après chaque exemple, une section **STATISTIQUES** s'affiche :

```
====================================================
  STATISTIQUES DE L'ONTOLOGIE BACK2SPEAK
====================================================
  Patients                 : 1
  Exercices en cours       : 1
  Exercices suivants       : 2    ← ceux de l'ontologie de base
  Phonèmes travail         : 1
  Phonèmes précédents      : 1
  Phonèmes suivants        : 0
  Catégories de mots       : 1
====================================================
```

Les chiffres des "exercices suivants" et "phonèmes" incluent les individus
déjà présents dans l'ontologie de base (`Exo2`, `Exo4`, `ph1`, `ph2`).

---

## Fichier de sortie

L'ontologie enrichie est sauvegardée dans :
```
ontologie/output/ontologie_enrichie.owl
```

Ce fichier peut être ouvert avec **Protégé** (logiciel gratuit d'édition d'ontologies)
pour visualiser graphiquement le résultat.

---

## En cas d'erreur

**`FileNotFoundError: Fichier ontologie introuvable`**
→ Vérifiez que vous lancez le script depuis le dossier `ontologie/`
→ Ou modifiez la variable `ONTOLOGY_PATH` dans le fichier

**`ImportError: Le module 'owlready2' est requis`**
→ Installez owlready2 : `pip install owlready2`

**`ModuleNotFoundError: No module named 'src'`**
→ Lancez le script depuis le dossier `ontologie/`, pas depuis `examples/`
