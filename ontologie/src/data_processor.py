"""
DataProcessor - Traitement des fichiers CSV du projet Back2speaK

Ce module lit les fichiers de données et prépare les informations
pour alimenter l'ontologie OWL.

Fichiers supportés :
    - audio_db.csv    : données patients (audio_id, speaker, age, sexe, ...)
    - exercices_a_realiser.csv : métadonnées phonétiques des items

Utilisation :
    from src.data_processor import load_patients_from_csv, process_batch_from_csv
"""

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional

from .ontology_populator import OntologyPopulator, PatientInfo


# ==============================================================================
# Structure de données ItemMetadata
# ==============================================================================

@dataclass
class ItemMetadata:
    """
    Métadonnées d'un item phonétique extrait du fichier exercices_a_realiser.csv.

    Attributs:
        item_id           : Identifiant de l'item, ex. "W_I01"
        mot               : Mot ou stimulus, ex. "Chat"
        type_item         : "Phonème", "syllabe" ou "Mot"
        position          : "Initiale", "Finale", "Médiane" ou "Isolée"
        nb_phonemes       : Nombre de phonèmes dans le mot
        forme             : Structure syllabique, ex. "CV", "CVC"
        voyelle_precedente: Voyelle avant le phonème cible
        classe_voy_prec   : Classification de la voyelle précédente
        voyelle_suivante  : Voyelle après le phonème cible
        classe_voy_suiv   : Classification de la voyelle suivante
    """
    item_id: str
    mot: str
    type_item: str
    position: str
    nb_phonemes: int
    forme: str
    voyelle_precedente: str
    classe_voy_prec: str
    voyelle_suivante: str
    classe_voy_suiv: str


# ==============================================================================
# Chargement des patients
# ==============================================================================

def load_patients_from_csv(csv_path: str) -> Dict[str, PatientInfo]:
    """
    Charge les informations patients depuis un fichier CSV.

    Le CSV doit contenir les colonnes :
        audio_id, collecteur_id, speaker, age, sexe, position, type_item, decision

    Un même patient peut apparaître plusieurs fois (un par enregistrement audio).
    Cette fonction déduplique par speaker et retourne un patient unique par speaker.

    Args:
        csv_path: Chemin vers le fichier CSV

    Returns:
        Dictionnaire {speaker_id: PatientInfo}

    Exemple:
        patients = load_patients_from_csv("Donnees/ch/audio_db.csv")
        patient_s02 = patients["S02"]
        print(patient_s02.age)  # 45
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    patients: Dict[str, PatientInfo] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker = row.get("speaker", "").strip()
            if not speaker or speaker in patients:
                continue

            try:
                age = int(row.get("age", 0))
            except (ValueError, TypeError):
                age = 0

            patient_id = f"{row.get('collecteur_id', '').strip()}_{speaker}"

            patients[speaker] = PatientInfo(
                patient_id=patient_id,
                speaker_id=speaker,
                age=age,
                sexe=row.get("sexe", "?").strip(),
                collecteur_id=row.get("collecteur_id", "?").strip(),
            )

    print(f"[OK] {len(patients)} patients chargés depuis : {csv_path}")
    return patients


# ==============================================================================
# Chargement des métadonnées phonétiques
# ==============================================================================

def load_items_metadata(csv_path: str) -> Dict[str, ItemMetadata]:
    """
    Charge les métadonnées des items phonétiques depuis le fichier CSV.

    Args:
        csv_path: Chemin vers exercices_a_realiser.csv

    Returns:
        Dictionnaire {item_id: ItemMetadata}

    Exemple:
        items = load_items_metadata("Donnees/ch/exercices_a_realiser.csv")
        item = items["W_I01"]
        print(item.mot)       # "Chat"
        print(item.position)  # "Initiale"
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    items: Dict[str, ItemMetadata] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("item_id", "").strip()
            if not item_id:
                continue

            try:
                nb_phonemes = int(row.get("Nombre de phonèmes", 0))
            except (ValueError, TypeError):
                nb_phonemes = 0

            items[item_id] = ItemMetadata(
                item_id=item_id,
                mot=row.get("mot / stimulus", "").strip(),
                type_item=row.get("type", "").strip(),
                position=row.get("position_du_/ʃ/", "").strip(),
                nb_phonemes=nb_phonemes,
                forme=row.get("Forme", "").strip(),
                voyelle_precedente=row.get("Voyelle_précédente", "").strip(),
                classe_voy_prec=row.get("Classe_voyelle_précédente", "").strip(),
                voyelle_suivante=row.get("voyelle_suivante", "").strip(),
                classe_voy_suiv=row.get("classe_voyelle_suivante", "").strip(),
            )

    print(f"[OK] {len(items)} items chargés depuis : {csv_path}")
    return items


# ==============================================================================
# Traitement par lot
# ==============================================================================

def _extract_speaker_from_audio_id(audio_id: str) -> Optional[str]:
    """
    Extrait le speaker_id depuis un audio_id.

    Exemples :
        "P02_S02_ISO01.wav" → "S02"
        "P02_S02_W_I01.wav" → "S02"
        "S02_test.wav"      → None (format non reconnu)
    """
    # Supprimer l'extension
    name = audio_id.replace(".wav", "").replace(".WAV", "")
    parts = name.split("_")

    # Format attendu: COLLECTEUR_SPEAKER_ITEM (ex: P02_S02_ISO01)
    # Le speaker est la 2ème partie et commence par "S"
    if len(parts) >= 2 and parts[1].upper().startswith("S"):
        return parts[1]

    return None


def process_batch_from_csv(
    populator: OntologyPopulator,
    label_csv_path: str,
    model_outputs: Dict[str, str],
    verbose: bool = True,
) -> dict:
    """
    Traite un lot d'enregistrements audio et leurs erreurs phonétiques.

    Cette fonction :
    1. Charge les patients depuis le CSV
    2. Pour chaque audio_id, identifie le patient correspondant
    3. Parse la sortie du modèle
    4. Ajoute chaque erreur dans l'ontologie

    Args:
        populator      : Instance d'OntologyPopulator chargée
        label_csv_path : Chemin vers le CSV des patients (audio_db.csv)
        model_outputs  : Dictionnaire {audio_id: texte_sortie_modele}
        verbose        : Afficher la progression (défaut: True)

    Returns:
        Dictionnaire de résultats avec les compteurs de traitement

    Exemple:
        model_outputs = {
            "P02_S02_ISO01.wav": \"\"\"
                Reference IPA (correct):   flɛʃ
                Produced IPA  (detected):  flɛs
                Error 1 of 1:
                Expected phoneme:   [ʃ]
                Produced phoneme:   [s]
                Position in word:   final (end of word)
                Preceded by:        [ɛ]
                Followed by:        [(none)]
            \"\"\"
        }

        results = process_batch_from_csv(
            populator,
            "Donnees/ch/audio_db.csv",
            model_outputs
        )
        print(f"Erreurs ajoutées : {results['errors_added']}")
    """
    patients = load_patients_from_csv(label_csv_path)

    stats = {
        "files_processed": 0,
        "files_skipped":   0,
        "errors_added":    0,
        "errors_skipped":  0,
    }

    for audio_id, model_text in model_outputs.items():
        speaker = _extract_speaker_from_audio_id(audio_id)

        if speaker is None or speaker not in patients:
            if verbose:
                print(f"  [SKIP] Patient introuvable pour : {audio_id}")
            stats["files_skipped"] += 1
            continue

        patient_info = patients[speaker]
        errors = populator.parse_model_output(model_text)

        if not errors:
            if verbose:
                print(f"  [INFO] Aucune erreur détectée dans : {audio_id}")
            stats["files_processed"] += 1
            continue

        for error in errors:
            try:
                populator.add_error_from_model_output(patient_info, error)
                stats["errors_added"] += 1
            except Exception as e:
                if verbose:
                    print(f"  [ERREUR] {audio_id} : {e}")
                stats["errors_skipped"] += 1

        stats["files_processed"] += 1

        if verbose:
            print(f"  [OK] {audio_id} → {len(errors)} erreur(s) ajoutée(s)")

    if verbose:
        print(f"\nRésumé du traitement :")
        print(f"  Fichiers traités : {stats['files_processed']}")
        print(f"  Fichiers ignorés : {stats['files_skipped']}")
        print(f"  Erreurs ajoutées : {stats['errors_added']}")
        print(f"  Erreurs ignorées : {stats['errors_skipped']}")

    return stats


# ==============================================================================
# Utilitaire : créer des erreurs depuis les métadonnées items
# ==============================================================================

def build_errors_from_items(
    items: Dict[str, ItemMetadata],
    phoneme_travail: str = "ʃ",
    filter_type: Optional[str] = None,
) -> List[dict]:
    """
    Construit une liste d'erreurs factices à partir des métadonnées items.

    Utile pour peupler l'ontologie avec tous les scénarios possibles
    (toutes les positions, tous les contextes) sans avoir de sorties modèle.

    Args:
        items          : Dictionnaire des items (depuis load_items_metadata)
        phoneme_travail: Phonème cible, ex. 'ʃ' (défaut)
        filter_type    : Filtrer par type ("Mot", "syllabe", etc.) ou None

    Returns:
        Liste de dictionnaires d'erreurs

    Exemple:
        items = load_items_metadata("exercices_a_realiser.csv")
        errors = build_errors_from_items(items, filter_type="Mot")
        # errors contient un dict par item de type "Mot"
    """
    errors = []

    for item in items.values():
        if filter_type and item.type_item.lower() != filter_type.lower():
            continue

        # Sauter les items isolés (pas de contexte)
        if item.position.lower() in ("isolée", "isolee", "isolated"):
            continue

        error = {
            "expected_phoneme": phoneme_travail,
            "produced_phoneme": "?",         # inconnu sans sortie modèle
            "position":         item.position,
            "preceded_by":      item.voyelle_precedente or "(none)",
            "followed_by":      item.voyelle_suivante or "(none)",
            "reference_ipa":    item.mot,
            "produced_ipa":     "",
        }
        errors.append(error)

    return errors
