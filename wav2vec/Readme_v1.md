# README — Analyseur IA de prononciation française

## Objectif du projet

Ce projet vise à analyser la prononciation d’un **mot** ou d’une **phrase** en français à partir d’un fichier audio `.wav`, puis à décider si le ou les mots sont **validés** ou **non validés**.

Le système ne se contente pas de faire une transcription phonémique. Il essaie aussi de fournir un retour exploitable pour un usage proche de l’orthophonie assistée :

* détection du mot problématique,
* détection du phonème concerné,
* distinction entre erreur claire et simple doute acoustique,
* détection d’un mot écrasé, peu net, incomplet ou avalé,
* retour articulatoire estimé, par exemple :

  * langue trop avancée,
  * langue trop reculée,
  * nasalisation insuffisante,
  * articulation faible.

---

## But métier

Le but métier est simple :

* un mot doit être **bien prononcé** pour être **validé**,
* un mot ne doit pas être refusé pour un simple doute acoustique léger si la reconnaissance principale est très confiante,
* quand un mot n’est pas validé, l’API doit expliquer **pourquoi** de façon claire.

Le système distingue donc :

* les **erreurs dures** : phonème remplacé, manquant, ajouté,
* les **doutes acoustiques légers** : phonème globalement reconnu mais un peu peu net,
* les **problèmes de qualité articulatoire** : mot écrasé, fin de mot peu nette, nasalisation douteuse.

---

## Vue d’ensemble du fonctionnement

Le pipeline suit les étapes suivantes :

1. réception du texte attendu et du fichier audio,
2. phonémisation du texte attendu,
3. décodage audio → phonèmes avec un modèle IA,
4. alignement entre phonèmes attendus et phonèmes prédits,
5. analyse phonème par phonème,
6. contrôle acoustique secondaire sur certains phonèmes sensibles,
7. diagnostic par mot,
8. décision finale : mot validé / non validé,
9. génération d’un rapport détaillé.

---

## Endpoint API

Le projet expose un seul endpoint principal :

* `POST /analyze`

Il accepte :

* `input_text` : mot ou phrase attendue,
* `audio_file` : fichier audio à analyser.

Le système détecte automatiquement si `input_text` contient :

* un **mot**,
* ou une **phrase**.

Il n’y a donc pas besoin d’avoir deux endpoints distincts.

---

## Architecture générale

Le système combine **3 niveaux de décision**.

### 1. Niveau phonémique principal

Le modèle IA principal convertit l’audio en phonèmes.

C’est la source principale de vérité pour savoir :

* quel phonème a été entendu,
* avec quelle confiance,
* quelles alternatives étaient possibles.

### 2. Niveau acoustique secondaire

Sur certains phonèmes sensibles, une vérification acoustique est ajoutée.

Exemples :

* `ʃ` vs `s`
* `ʒ` vs `z`

Cette étape sert à produire :

* un doute acoustique,
* un indice de mauvaise position de langue,
* un retour articulatoire plus fin.

Important : ce niveau secondaire ne doit pas écraser aveuglément le niveau principal. Il sert surtout de **signal complémentaire**.

### 3. Niveau métier

Le niveau métier décide :

* si le mot est **validé**,
* s’il reste **non validé**,
* et quel message doit être montré à l’utilisateur.

Un simple doute acoustique léger ne doit pas invalider un mot si la reconnaissance principale est très sûre.

---

## Briques utilisées

### 1. `phonemizer`

Utilisé pour convertir le texte attendu en phonèmes.

Configuration retenue :

* backend : `espeak`
* langue : `fr-fr`
* séparateur explicite entre phonèmes et mots

Exemple :

* `vache` → `v a ʃ`
* `la fourchette tombe` → `l a | f u ʁ ʃ ɛ t | t ɔ̃ b`

### 2. `eSpeak NG`

Nécessaire pour faire fonctionner `phonemizer` en local sur Windows.

### 3. `transformers` + `torch`

Utilisés pour charger le modèle de phonémisation audio.

### 4. Modèle audio principal

Le modèle utilisé est :

* `Cnam-LMSSC/wav2vec2-french-phonemizer`

Il sert à produire une séquence phonémique à partir de l’audio.

---

## Chargement et prétraitement audio

Le fichier audio envoyé à l’API est :

* lu avec `soundfile`,
* converti en mono si besoin,
* rééchantillonné en `16 kHz` si nécessaire.

### Choix retenus

* fréquence cible : **16 000 Hz**
* format interne : `float32`
* moyenne des canaux si stéréo

Ce choix est cohérent avec le modèle utilisé.

---

## Phonémisation du texte attendu

Le texte est normalisé :

* trim,
* passage en minuscules,
* normalisation des espaces.

Ensuite, il est converti en phonèmes avec `phonemizer`.

### Pourquoi cette étape est essentielle

Elle permet d’obtenir la **suite de phonèmes attendue**, utilisée ensuite pour :

* l’alignement,
* la comparaison phonème par phonème,
* le découpage mot → phonèmes,
* le diagnostic local.

---

## Décodage audio → phonèmes

Le modèle audio renvoie des logits CTC.

Le système :

* applique un `argmax`,
* collapse les répétitions CTC,
* enlève le blank token,
* calcule la **confiance moyenne** du token retenu,
* calcule aussi les **alternatives probables**.

Chaque phonème prédit possède donc :

* un token,
* une confiance,
* une plage temporelle estimée,
* une énergie locale,
* une liste d’alternatives.

---

## Fusion des voyelles nasales

Une correction importante a été ajoutée pour les voyelles nasales.

Le modèle audio peut parfois produire :

* `ɔ`
* puis `̃`

alors que, du point de vue phonologique, on attend :

* `ɔ̃`

Le programme fusionne donc automatiquement :

* `ɑ` + `̃` → `ɑ̃`
* `ɛ` + `̃` → `ɛ̃`
* `ɔ` + `̃` → `ɔ̃`
* `œ` + `̃` → `œ̃`

### Pourquoi

Sans cette fusion, des mots comme `tombe` pouvaient produire de faux positifs du type :

* son ajouté,
* nasalisation cassée,
* faux mismatch.

---

## Alignement phonémique

Le système utilise un alignement de type **Levenshtein** entre :

* la séquence attendue,
* la séquence prédite.

Cela permet de détecter :

* `equal`
* `replace`
* `delete`
* `insert`

Ces opérations sont ensuite converties en diagnostics métier :

* `correct`
* `mismatch`
* `deleted`
* `inserted`
* `uncertain`

---

## Statuts possibles par phonème

Chaque phonème analysé reçoit un statut.

### `correct`

Le phonème attendu et le phonème prédit correspondent, sans alerte notable.

### `uncertain`

Le phonème semble correct, mais un doute subsiste.

Ce doute peut venir de :

* confiance faible,
* énergie trop faible,
* contrôle acoustique secondaire contradictoire.

### `mismatch`

Le phonème produit ne correspond pas au phonème attendu.

### `deleted`

Le phonème attendu semble absent.

### `inserted`

Un son supplémentaire apparaît.

---

## Contrôle acoustique secondaire

Le programme ajoute un contrôle acoustique secondaire pour certains phonèmes sensibles :

* `ʃ`
* `s`
* `ʒ`
* `z`

### Métrique utilisée

Le système utilise principalement le **spectral centroid**.

Le spectral centroid mesure le centre de gravité du spectre fréquentiel :

* plus il est haut, plus le son est riche en hautes fréquences,
* plus il est bas, plus l’énergie est concentrée dans les fréquences basses/médiums.

### Utilité

Il est utile notamment pour des oppositions comme :

* `ʃ` vs `s`
* `ʒ` vs `z`

Exemple :

* si le phonème attendu est `ʃ`,
* mais que le centroid suggère un comportement plus proche de `s`,
* alors le système peut créer un **doute acoustique**.

### Règle importante actuelle

Le contrôle acoustique secondaire **ne doit pas à lui seul invalider un mot** si :

* le phonème principal est reconnu correctement,
* avec très haute confiance,
* et qu’il n’y a pas d’erreur dure sur le mot.

Autrement dit :

* il peut générer un **warning**,
* mais pas forcément un **échec métier**.

---

## Feedback articulatoire estimé

Le projet ajoute un retour `articulatory_feedback` pour certains cas.

Important :

* la position de langue n’est **pas directement observée**,
* elle est **estimée** à partir des indices acoustiques.

### Exemples de feedback possibles

* `tongue_position = trop_avant`
* `tongue_position = trop_reculée`
* `airflow = frication_trop_aigue`
* `nasalization = insuffisante`
* `strength = faible`

### Exemple métier

Pour un `ʃ` qui tire vers `s`, le système peut produire :

* langue trop avancée,
* son trop aigu,
* le son se rapproche de “s” au lieu de “ch”.

---

## Détection des mots écrasés

Le système essaie aussi de repérer si un mot est :

* écrasé,
* trop court,
* trop faible,
* peu articulé.

### Métriques utilisées

#### `duration_sec`

Durée estimée du mot.

#### `mean_energy`

Énergie moyenne du mot.

#### `clarity_score`

Score de clarté du mot.

#### `completeness_score`

Score de complétude du mot.

### Heuristique de mot écrasé

Un mot peut être considéré comme écrasé si :

* sa durée par phonème est trop faible,
* et son énergie moyenne est trop faible,
* ou sa clarté globale est trop basse,
* ou le mot est incomplet.

---

## Tolérance métier sur les doutes acoustiques

Une règle importante a été ajoutée pour éviter les faux positifs.

### Cas toléré

Si un mot contient :

* un seul phonème `uncertain`,
* mais que ce phonème est bien reconnu,
* et que sa confiance est très élevée,
* et qu’il n’y a aucune erreur dure,

alors le mot reste **validé**.

### Exemple typique

* `expected = ʃ`
* `predicted = ʃ`
* `confidence = 0.9998`
* doute acoustique léger vers `s`

Dans ce cas, le mot doit rester valide, avec au plus :

* un warning interne,
* un léger doute technique,
* mais pas une invalidation métier.

---

## Sortie JSON

L’API retourne plusieurs blocs utiles.

### 1. `phoneme_analysis`

Analyse détaillée phonème par phonème.

### 2. `acoustic_checks`

Contrôle secondaire sur les phonèmes sensibles.

### 3. `words`

Diagnostic détaillé par mot.

Chaque mot contient notamment :

* `is_valid`
* `clarity_score`
* `completeness_score`
* `is_crushed`
* `problem_positions`
* `warnings`

### 4. `word_reports`

Résumé métier clair, prêt à afficher.

Chaque entrée contient :

* `word`
* `is_valid`
* `main_reason`
* `message`
* `details`

### 5. `final_decision`

Décision globale finale :

* `valid`
* `invalid`

### 6. `summary`

Résumé quantitatif :

* substitutions,
* deletions,
* insertions,
* uncertain,
* mots écrasés,
* score global.

---

## Logique de validation métier

### Un mot est non validé si

* il contient une vraie erreur dure,
* ou il est écrasé,
* ou il contient un doute important non tolérable.

### Un mot reste validé si

* aucun phonème n’est erroné,
* le mot n’est pas écrasé,
* et un éventuel doute acoustique est léger et isolé.

### Décision finale

* si au moins un mot est non validé → `final_decision = invalid`
* sinon → `final_decision = valid`

---

## Paramètres actuellement configurés

### Audio

* `TARGET_SAMPLING_RATE = 16000`

### Confiance

* `CONFIDENCE_THRESHOLD = 0.65`
* `LOW_CONFIDENCE_THRESHOLD = 0.50`
* `HIGH_CONFIDENCE_VALIDATION_THRESHOLD = 0.98`

### Durée / énergie

* `MIN_WORD_DURATION_PER_PHONEME = 0.055`
* `LOW_ENERGY_THRESHOLD = 2e-5`
* `VERY_LOW_ENERGY_THRESHOLD = 8e-6`

### Contrôle acoustique secondaire

* `SH_S_CENTROID_THRESHOLD = 3600.0`
* `ZH_Z_CENTROID_THRESHOLD = 3300.0`
* `ACOUSTIC_DISAGREE_THRESHOLD = 0.55`

### Phonèmes sensibles

* `ʃ`
* `s`
* `ʒ`
* `z`

---

## Choix techniques importants

### Un seul endpoint

Cela simplifie :

* l’intégration front,
* les tests,
* la maintenance.

### Fusion des nasales

Ce choix réduit fortement les faux positifs sur les voyelles nasales françaises.

### Heuristique acoustique secondaire non dominante

Le modèle principal garde la priorité.

### Rapport métier séparé du debug

Le système garde :

* un niveau technique détaillé,
* un niveau métier clair pour l’affichage.

---

## Exemples de diagnostics métier attendus

### Cas 1 — mot correct

```json
{
  "word": "tombe",
  "is_valid": true,
  "message": "Le mot \"tombe\" semble correct."
}
```

### Cas 2 — mot non validé, phonème remplacé

```json
{
  "word": "vache",
  "is_valid": false,
  "main_reason": "phoneme_remplace",
  "message": "Le mot \"vache\" n'est pas validé : un phonème semble remplacé."
}
```

### Cas 3 — mot non validé, fin de mot peu nette

```json
{
  "word": "fourchette",
  "is_valid": false,
  "main_reason": "fin_de_mot_peu_nette",
  "message": "Le mot \"fourchette\" n'est pas validé : la fin du mot est peu nette."
}
```

### Cas 4 — mot validé malgré léger doute acoustique

```json
{
  "word": "fourchette",
  "is_valid": true,
  "warnings": ["léger doute acoustique"]
}
```

---

## Installation

### Pré-requis Windows

* Windows 10/11
* Python 3.10 ou 3.11
* eSpeak NG installé

### Création de l’environnement virtuel

```powershell
cd D:\Dataton2026
mkdir pronunciation_analyzer
cd pronunciation_analyzer
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloque l’activation :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Mise à jour de pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Installation des dépendances Python

```powershell
pip install fastapi uvicorn soundfile numpy torch transformers phonemizer python-multipart
```

---

## Installation de eSpeak NG

Installer **eSpeak NG** sur Windows.

Chemin recommandé :

```text
C:\Program Files\eSpeak NG
```

Le dossier doit contenir au minimum :

* `espeak-ng.exe`
* `libespeak-ng.dll`

Le programme tente de les détecter automatiquement.

---

## Lancement de l’API

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Puis ouvrir :

```text
http://127.0.0.1:8000/docs
```

---

## Utilisation

### Requête

* `input_text` : mot ou phrase attendue
* `audio_file` : fichier `.wav`

### Exemple

* texte : `la fourchette tombe`
* audio : fichier `.wav`

Le système retournera :

* les phonèmes attendus,
* les phonèmes prédits,
* le détail par phonème,
* le détail par mot,
* la décision métier finale,
* les explications articulatoires estimées.

---

## Lecture recommandée côté frontend

### Pour l’affichage utilisateur

Afficher en priorité :

1. `final_decision`
2. `word_reports`
3. `words[].warnings`

### Pour le debug développeur

Lire :

1. `phoneme_analysis`
2. `acoustic_checks`
3. `summary`

---

## Limites actuelles

Le projet reste un prototype avancé. Il a encore des limites :

1. le découpage temporel des phonèmes reste approximatif,
2. le contrôle acoustique secondaire repose sur des heuristiques,
3. la position de langue est estimée indirectement,
4. certains contrastes phonétiques demanderaient un classifieur spécialisé,
5. le système n’effectue pas encore de forced alignment précis.

---

## Améliorations recommandées

### 1. Forced alignment

Pour mieux localiser les phonèmes dans le temps.

### 2. Classifieurs spécialisés

Exemples :

* `/ʃ/ vs /s/`
* `/ʒ/ vs /z/`
* `/ʁ/ vs /l/`

### 3. Feedback pédagogique court

Exemples :

* “Recule un peu la langue pour mieux faire le son ch.”
* “Marque davantage la fin du mot.”
* “Ajoute mieux la nasalisation.”

### 4. Modes de difficulté

Exemples :

* débutant,
* normal,
* strict.

### 5. Historique de progression

Suivi du score par mot, phonème et séance.

---

## Résumé final

Ce projet est un analyseur IA de prononciation française qui :

* reçoit un texte attendu et un audio,
* convertit le texte en phonèmes,
* convertit l’audio en phonèmes,
* compare les deux,
* détecte les écarts,
* tolère les petits doutes acoustiques quand le mot est clairement correct,
* explique pourquoi un mot n’est pas validé,
* et fournit un retour articulatoire estimé proche d’un usage orthophonique.

C’est une base sérieuse pour construire un assistant de prononciation français plus intelligent, plus pédagogique et plus robuste.
