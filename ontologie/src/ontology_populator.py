"""
OntologyPopulator - Module principal Back2speaK

Peuple automatiquement une ontologie OWL avec les résultats de détection
d'erreurs phonétiques produits par le modèle d'analyse.

Utilisation rapide:
    from src.ontology_populator import OntologyPopulator, PatientInfo

    populator = OntologyPopulator("ontologie.owx")

    patient = PatientInfo("P02_S02", "S02", 45, "F", "P02")

    error = {
        'expected_phoneme': 'ʃ',
        'produced_phoneme': 's',
        'position': 'final',
        'preceded_by': 'ɛ',
        'followed_by': '(none)',
    }

    populator.add_error_from_model_output(patient, error)
    populator.save_ontology("output/ontologie_enrichie.owl")
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from owlready2 import get_ontology, World
except ImportError:
    raise ImportError(
        "Le module 'owlready2' est requis.\n"
        "Installez-le avec : pip install owlready2"
    )


# ==============================================================================
# Structure de données Patient
# ==============================================================================

@dataclass
class PatientInfo:
    """
    Informations sur un patient.

    Attributs:
        patient_id    : Identifiant unique, ex. "P02_S02"
        speaker_id    : Identifiant du locuteur, ex. "S02"
        age           : Âge du patient (en mois pour les enfants)
        sexe          : "F" (féminin) ou "M" (masculin)
        collecteur_id : Identifiant du collecteur/orthophoniste, ex. "P02"
    """
    patient_id: str
    speaker_id: str
    age: int
    sexe: str
    collecteur_id: str


# ==============================================================================
# Classe principale
# ==============================================================================

class OntologyPopulator:
    """
    Peuple une ontologie OWL avec des résultats de détection phonétique.

    Cette classe :
    1. Charge une ontologie OWL existante (.owx / .owl)
    2. Crée automatiquement les patients, exercices et phonèmes
    3. Respecte les règles orthophoniques (niveaux selon position)
    4. Utilise des caches pour éviter les doublons
    5. Sauvegarde l'ontologie enrichie

    Règles de niveaux de difficulté :
        Position "Debut"  (initial) = Niveau 3 (plus difficile)
        Position "Fin"    (final)   = Niveau 2 (difficile)
        Position "Milieu" (médian)  = Niveau 1 (moins difficile)
    """

    # ------------------------------------------------------------------
    # Tables de correspondance
    # ------------------------------------------------------------------

    # Normalisation des positions vers les valeurs de l'ontologie
    _POSITION_MAP: Dict[str, str] = {
        # Français
        "initiale": "Debut",
        "initial":  "Debut",
        "début":    "Debut",
        "debut":    "Debut",
        "première": "Debut",
        "premiere": "Debut",
        # Anglais
        "start":     "Debut",
        "beginning": "Debut",
        # Fin / Final
        "finale":  "Fin",
        "final":   "Fin",
        "fin":     "Fin",
        "end":     "Fin",
        # Milieu / Médian
        "médiane":  "Milieu",
        "mediane":  "Milieu",
        "médian":   "Milieu",
        "median":   "Milieu",
        "medial":   "Milieu",
        "milieu":   "Milieu",
        "middle":   "Milieu",
        "centrale": "Milieu",
        "central":  "Milieu",
        "interne":  "Milieu",
    }

    # Niveau de difficulté selon la position
    _LEVEL_MAP: Dict[str, int] = {
        "Debut":  3,
        "Fin":    2,
        "Milieu": 1,
    }

    # Phonèmes IPA → type dans l'ontologie (aTypeTravail)
    # L'ontologie Back2speaK supporte "Ch" (/ʃ/) et "Ze" (/ʒ/)
    _PHONEME_TYPE_MAP: Dict[str, str] = {
        "ʃ": "Ch",
        "ʒ": "Ze",
        # Variantes transcription
        "sh": "Ch",
        "zh": "Ze",
        "ch": "Ch",
        "ze": "Ze",
    }

    # Valeurs vides/absentes pour les contextes phonémiques
    _NONE_VALUES = {"(none)", "none", "-", "", "aucun", "aucune", "–", "—"}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, ontology_path: str):
        """
        Charge l'ontologie depuis un fichier .owx ou .owl.

        Args:
            ontology_path: Chemin vers le fichier ontologie

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        abs_path = os.path.abspath(ontology_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Fichier ontologie introuvable : {abs_path}\n"
                "Vérifiez le chemin fourni."
            )

        # owlready2 accepte le chemin Windows directement (C:/...) ou POSIX (/...)
        # On utilise des slashes forward pour la compatibilité
        load_path = abs_path.replace("\\", "/")

        self._world = World()
        self.onto = self._world.get_ontology(load_path).load()
        self.base_iri = self.onto.base_iri

        # Caches : évitent de créer plusieurs fois le même individu
        self._patients:   Dict[str, object] = {}  # patient_id → individu
        self._ph_travail: Dict[str, object] = {}  # "phoneme_position" → individu
        self._ph_context: Dict[str, object] = {}  # "class_phoneme" → individu
        self._categories: Dict[str, object] = {}  # "phoneme_position" → individu
        self._exercises:  Dict[str, object] = {}  # "patient_phoneme_pos" → individu

        # Compteurs pour garantir des noms uniques
        self._name_counters: Dict[str, int] = {}

        print(f"[OK] Ontologie chargée : {ontology_path}")
        print(f"     IRI de base : {self.base_iri}")

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------

    def _sanitize(self, text: str) -> str:
        """
        Convertit une chaîne en nom valide pour OWL.
        Remplace les symboles IPA et les caractères spéciaux.
        """
        replacements = {
            "ʃ": "SH",  "ʒ": "ZH",  "ɛ": "E_ouv", "œ": "EU",
            "ø": "EU_f", "ɔ": "O_ouv", "ɑ": "A_arr",
            "ɑ̃": "AN",  "ɛ̃": "EIN", "ɔ̃": "ON",  "œ̃": "UN",
            "ã": "AN",   "ẽ": "EIN",  "õ": "ON",
            "ə": "schwa", "ɥ": "HY",  "ʁ": "R",
            "ɲ": "GN",   "ŋ": "NG",
        }
        result = text
        for ipa, ascii_rep in replacements.items():
            result = result.replace(ipa, ascii_rep)
        # Remplacer tous les caractères restants non alphanumériques
        return re.sub(r"[^a-zA-Z0-9_]", "_", result)

    def _unique_name(self, base: str) -> str:
        """
        Génère un nom unique dans l'ontologie.
        Si 'base' est déjà utilisé, ajoute un suffixe numérique.
        """
        if self.onto[base] is None:
            return base
        count = self._name_counters.get(base, 1) + 1
        while True:
            candidate = f"{base}_{count}"
            if self.onto[candidate] is None:
                self._name_counters[base] = count
                return candidate
            count += 1

    def _normalize_position(self, position: str) -> str:
        """
        Normalise une position vers les valeurs de l'ontologie.

        Exemples :
            "initial"                    → "Debut"
            "final (end of word)"        → "Fin"
            "Médiane"                    → "Milieu"
            "Initiale"                   → "Debut"
        """
        key = position.lower().strip()

        # Vérification exacte
        if key in self._POSITION_MAP:
            return self._POSITION_MAP[key]

        # Vérification partielle : "initial (beginning of word)" etc.
        for k, v in self._POSITION_MAP.items():
            if key.startswith(k):
                return v

        print(f"  [ATTENTION] Position inconnue '{position}' → 'Milieu' par défaut")
        return "Milieu"

    def _map_phoneme_type(self, phoneme_ipa: str) -> Optional[str]:
        """
        Mappe un phonème IPA vers le type de l'ontologie ('Ch' ou 'Ze').
        Retourne None si le phonème n'est pas reconnu.
        """
        key = phoneme_ipa.lower().strip()
        return self._PHONEME_TYPE_MAP.get(key, None)

    def _is_none_value(self, value: str) -> bool:
        """Vérifie si une valeur représente une absence (none, -, etc.)."""
        return value.lower().strip() in self._NONE_VALUES

    # ------------------------------------------------------------------
    # Création des entités OWL
    # ------------------------------------------------------------------

    def _get_or_create_patient(self, info: PatientInfo) -> object:
        """
        Récupère ou crée un individu Patient dans l'ontologie.
        Utilise le cache pour éviter les doublons.
        """
        if info.patient_id in self._patients:
            return self._patients[info.patient_id]

        with self.onto:
            Patient = self.onto.Patient
            name = self._unique_name(self._sanitize(info.patient_id))
            patient = Patient(name)

        self._patients[info.patient_id] = patient
        return patient

    def _get_or_create_ph_travail(self, phoneme_ipa: str,
                                   position: str) -> object:
        """
        Récupère ou crée un phTravail (phonème cible de l'exercice).

        Le cache combine phonème + position pour permettre le même phonème
        travaillé à différentes positions (ex: ʃ en début vs en fin).
        """
        cache_key = f"{phoneme_ipa}__{position}"
        if cache_key in self._ph_travail:
            return self._ph_travail[cache_key]

        ph_type = self._map_phoneme_type(phoneme_ipa)

        with self.onto:
            phTravail = self.onto.phTravail
            safe = self._sanitize(phoneme_ipa)
            name = self._unique_name(f"phT_{safe}_{self._sanitize(position)}")
            ph = phTravail(name)

            ph.aPosition.append(position)
            if ph_type is not None:
                ph.aTypeTravail.append(ph_type)

        self._ph_travail[cache_key] = ph
        return ph

    def _get_or_create_ph_context(self, phoneme_ipa: str,
                                   ph_class_name: str) -> object:
        """
        Récupère ou crée un phonème de contexte (phPrecedent ou phSuivant).

        Ces phonèmes représentent l'environnement sonore du phonème travaillé.
        """
        cache_key = f"{ph_class_name}__{phoneme_ipa}"
        if cache_key in self._ph_context:
            return self._ph_context[cache_key]

        with self.onto:
            ph_class = getattr(self.onto, ph_class_name)
            prefix = "phPrec" if ph_class_name == "phPrecedent" else "phSuiv"
            name = self._unique_name(f"{prefix}_{self._sanitize(phoneme_ipa)}")
            ph = ph_class(name)
            ph.aType.append(phoneme_ipa)

        self._ph_context[cache_key] = ph
        return ph

    def _get_or_create_categorie(self, phoneme_ipa: str,
                                  position: str) -> object:
        """
        Récupère ou crée une CategorieMot pour un phonème à une position donnée.

        La CategorieMot regroupe les mots contenant le phonème cible
        à la même position dans le mot.
        """
        cache_key = f"{phoneme_ipa}__{position}"
        if cache_key in self._categories:
            return self._categories[cache_key]

        with self.onto:
            CategorieMot = self.onto.CategorieMot
            safe = self._sanitize(phoneme_ipa)
            name = self._unique_name(f"Cat_{safe}_{self._sanitize(position)}")
            cat = CategorieMot(name)

        self._categories[cache_key] = cat
        return cat

    def _get_or_create_exercise(self, patient: object, ph_travail: object,
                                 categorie: object, position: str,
                                 success_rate: float = 0.0) -> object:
        """
        Récupère ou crée un ExoEnCours pour un patient et un phonème.

        Le niveau est assigné automatiquement selon la position :
            Debut  → niveau 3
            Fin    → niveau 2
            Milieu → niveau 1

        Args:
            success_rate : taux de réussite observé (0.0 → 1.0).
                           Converti en entier 0-100 dans l'ontologie.
                           Si l'exercice existe déjà, met à jour le taux.
        """
        cache_key = f"{patient.name}__{ph_travail.name}"
        taux = max(0, min(100, int(round(success_rate * 100))))

        if cache_key in self._exercises:
            exo = self._exercises[cache_key]
            # Mettre à jour le taux si un nouveau calcul est disponible
            with self.onto:
                exo.aTauxReussite.clear()
                exo.aTauxReussite.append(taux)
            return exo

        niveau = self._LEVEL_MAP.get(position, 1)

        with self.onto:
            ExoEnCours = self.onto.ExoEnCours
            name = self._unique_name(f"Exo_{patient.name}_{ph_travail.name}")
            exo = ExoEnCours(name)
            exo.aNiveau.append(niveau)
            exo.aTauxReussite.append(taux)
            exo.aCategorieMots.append(categorie)

        self._exercises[cache_key] = exo
        return exo

    # ------------------------------------------------------------------
    # Méthode principale : ajout d'une erreur
    # ------------------------------------------------------------------

    def add_error_from_model_output(self, patient_info: PatientInfo,
                                     error: dict) -> None:
        """
        Ajoute une erreur phonétique détectée dans l'ontologie.

        Crée automatiquement (si pas déjà présents) :
        - Le patient
        - Le phonème de travail (phTravail) avec position et type
        - Les phonèmes de contexte (phPrecedent, phSuivant)
        - La catégorie de mots (CategorieMot)
        - L'exercice en cours (ExoEnCours) avec niveau et taux = 0
        - La liaison Patient → pratique → ExoEnCours

        Args:
            patient_info : Informations sur le patient
            error        : Dictionnaire avec les clés suivantes :
                - expected_phoneme (str)   : phonème attendu en IPA, ex. 'ʃ'
                - produced_phoneme (str)   : phonème produit en IPA, ex. 's'
                - position         (str)   : position dans le mot
                - preceded_by      (str)   : phonème précédent (IPA) ou '(none)'
                - followed_by      (str)   : phonème suivant (IPA) ou '(none)'
                - success_rate     (float) : taux de réussite 0.0–1.0 (optionnel, défaut=0)
                - reference_ipa    (str)   : mot de référence en IPA (optionnel)
                - produced_ipa     (str) : mot produit en IPA (optionnel)
        """
        # --- Normaliser la position ---
        position_raw = error.get("position", "medial")
        position = self._normalize_position(position_raw)

        expected   = error.get("expected_phoneme", "?")
        preceded   = error.get("preceded_by", "(none)")
        followed     = error.get("followed_by", "(none)")
        success_rate = float(error.get("success_rate", 0.0))

        with self.onto:
            # 1. Patient
            patient = self._get_or_create_patient(patient_info)

            # 2. Phonème de travail
            ph_travail = self._get_or_create_ph_travail(expected, position)

            # 3. Phonème précédent
            if not self._is_none_value(preceded):
                ph_prec = self._get_or_create_ph_context(preceded, "phPrecedent")
                if ph_prec not in ph_travail.aPrecedent:
                    ph_travail.aPrecedent.append(ph_prec)

            # 4. Phonème suivant
            if not self._is_none_value(followed):
                ph_suiv = self._get_or_create_ph_context(followed, "phSuivant")
                if ph_suiv not in ph_travail.aSuivant:
                    ph_travail.aSuivant.append(ph_suiv)

            # 5. Catégorie de mots
            categorie = self._get_or_create_categorie(expected, position)
            if ph_travail not in categorie.aPhoneme:
                categorie.aPhoneme.append(ph_travail)

            # 6. Exercice en cours (avec taux de réussite réel)
            exo = self._get_or_create_exercise(
                patient, ph_travail, categorie, position, success_rate
            )

            # 7. Lier patient → exercice
            if exo not in patient.pratique:
                patient.pratique.append(exo)

    # ------------------------------------------------------------------
    # Parsing de la sortie texte du modèle
    # ------------------------------------------------------------------

    def parse_model_output(self, text: str) -> List[dict]:
        """
        Parse le texte brut produit par le modèle de détection phonétique.

        Format attendu :
            Reference IPA (correct):   flɛʃ
            Produced IPA  (detected):  flɛs
            Total phonemes in reference word: 4
            Number of ʃ errors detected:      1
            Error 1 of 1:
            Expected phoneme:   [ʃ]
            Produced phoneme:   [s]
            Position in word:   final (end of word)
            Preceded by:        [ɛ]
            Followed by:        [(none)]

        Args:
            text: Texte complet produit par le modèle

        Returns:
            Liste de dictionnaires d'erreurs (un par erreur détectée)
        """
        errors = []

        # Extraire les IPA globaux
        ref_m  = re.search(r"Reference IPA.*?:\s*(\S+)", text)
        prod_m = re.search(r"Produced IPA.*?:\s*(\S+)",  text)
        ref_ipa  = ref_m.group(1).strip()  if ref_m  else ""
        prod_ipa = prod_m.group(1).strip() if prod_m else ""

        # Séparer les blocs d'erreurs
        blocks = re.split(r"Error\s+\d+\s+of\s+\d+\s*:", text)[1:]

        for block in blocks:
            expected_m  = re.search(r"Expected phoneme\s*:\s*\[([^\]]+)\]", block)
            produced_m  = re.search(r"Produced phoneme\s*:\s*\[([^\]]+)\]", block)
            position_m  = re.search(r"Position in word\s*:\s*(.+?)(?:\n|$)",  block)
            preceded_m  = re.search(r"Preceded by\s*:\s*\[([^\]]*)\]",        block)
            followed_m  = re.search(r"Followed by\s*:\s*\[([^\]]*)\]",        block)

            if not expected_m or not produced_m:
                continue

            errors.append({
                "expected_phoneme": expected_m.group(1).strip(),
                "produced_phoneme": produced_m.group(1).strip(),
                "position":   position_m.group(1).strip() if position_m else "medial",
                "preceded_by": preceded_m.group(1).strip() if preceded_m else "(none)",
                "followed_by": followed_m.group(1).strip() if followed_m else "(none)",
                "reference_ipa": ref_ipa,
                "produced_ipa":  prod_ipa,
            })

        return errors

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """
        Retourne des statistiques sur l'ontologie peuplée.

        Returns:
            Dictionnaire avec les compteurs de chaque type d'entité
        """
        return {
            "patients":            len(list(self.onto.Patient.instances())),
            "exercices_en_cours":  len(list(self.onto.ExoEnCours.instances())),
            "exercices_suivants":  len(list(self.onto.ExoSuivant.instances())),
            "phonemes_travail":    len(list(self.onto.phTravail.instances())),
            "phonemes_precedents": len(list(self.onto.phPrecedent.instances())),
            "phonemes_suivants":   len(list(self.onto.phSuivant.instances())),
            "categories_mots":     len(list(self.onto.CategorieMot.instances())),
        }

    def print_statistics(self) -> None:
        """Affiche les statistiques de manière lisible dans la console."""
        stats = self.get_statistics()
        print("\n" + "=" * 52)
        print("  STATISTIQUES DE L'ONTOLOGIE BACK2SPEAK")
        print("=" * 52)
        print(f"  Patients                 : {stats['patients']}")
        print(f"  Exercices en cours       : {stats['exercices_en_cours']}")
        print(f"  Exercices suivants       : {stats['exercices_suivants']}")
        print(f"  Phonèmes travail         : {stats['phonemes_travail']}")
        print(f"  Phonèmes précédents      : {stats['phonemes_precedents']}")
        print(f"  Phonèmes suivants        : {stats['phonemes_suivants']}")
        print(f"  Catégories de mots       : {stats['categories_mots']}")
        print("=" * 52 + "\n")

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save_ontology(self, output_path: str,
                      format: str = "rdfxml") -> None:
        """
        Sauvegarde l'ontologie enrichie dans un fichier.

        Args:
            output_path : Chemin de sortie, ex. "output/ontologie_enrichie.owl"
            format      : Format de sauvegarde.
                          'rdfxml' (défaut) → compatible Protégé / OWLAPI
                          'ntriples'        → format N-Triples
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.onto.save(file=output_path, format=format)
        print(f"[OK] Ontologie sauvegardée : {os.path.abspath(output_path)}")
