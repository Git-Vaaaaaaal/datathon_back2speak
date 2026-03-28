"""
Exemple d'utilisation du système Back2speaK Ontologie
======================================================

Ce fichier montre 4 façons d'utiliser le système :

    Exemple 1 : Ajouter une erreur manuellement (cas simple)
    Exemple 2 : Parser la sortie brute du modèle
    Exemple 3 : Traitement par lot depuis le CSV
    Exemple 4 : Explorer les items phonétiques

Pour lancer cet exemple depuis le dossier 'ontologie/' :
    python examples/exemple_utilisation.py

Prérequis :
    pip install owlready2
"""

import os
import sys

# Ajouter le dossier parent au chemin Python pour trouver le package 'src'
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.ontology_populator import OntologyPopulator, PatientInfo
from src.data_processor import (
    load_patients_from_csv,
    load_items_metadata,
    process_batch_from_csv,
    build_errors_from_items,
)

# ==============================================================================
# Chemins des fichiers
# ==============================================================================

# Ontologie de base (à charger)
ONTOLOGY_PATH = os.path.join(parent_dir, "ontologie.owx")

# Fichiers de données du projet
PROJECT_ROOT    = os.path.dirname(parent_dir)
LABEL_CSV_PATH  = os.path.join(PROJECT_ROOT, "Donnees", "ch", "audio_db.csv")
META_CSV_PATH   = os.path.join(PROJECT_ROOT, "Donnees", "ch", "exercices_a_realiser.csv")

# Fichier de sortie
OUTPUT_PATH = os.path.join(parent_dir, "output", "ontologie_enrichie.owl")


# ==============================================================================
# Exemple 1 : Ajout manuel d'une erreur
# ==============================================================================

def exemple_1_ajout_manuel():
    """
    Cas le plus simple : ajouter une erreur phonétique à la main.

    Scénario : le patient S02 (collecteur P02, 45 ans, F) prononce
    le mot 'flèche' (flɛʃ) comme 'flɛs' → erreur sur le phonème /ʃ/ en position finale.
    """
    print("\n" + "=" * 60)
    print("EXEMPLE 1 : Ajout manuel d'une erreur phonétique")
    print("=" * 60)

    # Charger l'ontologie
    populator = OntologyPopulator(ONTOLOGY_PATH)

    # Définir le patient
    patient = PatientInfo(
        patient_id="P02_S02",
        speaker_id="S02",
        age=45,
        sexe="F",
        collecteur_id="P02",
    )

    # Définir l'erreur détectée par le modèle
    error = {
        "expected_phoneme": "ʃ",        # phonème attendu
        "produced_phoneme": "s",         # phonème produit (incorrect)
        "position":         "final",     # position dans le mot
        "preceded_by":      "ɛ",         # phonème précédent
        "followed_by":      "(none)",    # pas de phonème suivant
        "reference_ipa":    "flɛʃ",     # mot de référence
        "produced_ipa":     "flɛs",     # mot produit
    }

    # Ajouter l'erreur dans l'ontologie
    populator.add_error_from_model_output(patient, error)

    # Afficher les statistiques
    populator.print_statistics()

    # Sauvegarder
    populator.save_ontology(OUTPUT_PATH)
    print("\nExemple 1 terminé.")


# ==============================================================================
# Exemple 2 : Parser la sortie brute du modèle
# ==============================================================================

def exemple_2_parse_modele():
    """
    Parser le texte exact produit par le modèle de détection phonétique.

    Ce format correspond à la sortie standard du modèle Back2speaK.
    """
    print("\n" + "=" * 60)
    print("EXEMPLE 2 : Parsing de la sortie du modèle")
    print("=" * 60)

    populator = OntologyPopulator(ONTOLOGY_PATH)

    # Texte exact produit par le modèle (copié/collé de la sortie)
    model_output_text = """
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
"""

    # Le modèle peut aussi détecter plusieurs erreurs dans un mot :
    model_output_multi = """
Reference IPA (correct):   ʃɑ̃tɛ
Produced IPA  (detected):  sɑntɛ
Total phonemes in reference word: 5
Number of ʃ errors detected:      1
Error 1 of 1:
Expected phoneme:   [ʃ]
Produced phoneme:   [s]
Position in word:   initial (beginning of word)
Preceded by:        [(none)]
Followed by:        [ɑ̃]
"""

    patient = PatientInfo("P02_S02", "S02", 45, "F", "P02")

    # Parser et ajouter la première sortie
    errors_1 = populator.parse_model_output(model_output_text)
    print(f"\nErreurs parsées (mot 1) : {len(errors_1)}")
    for e in errors_1:
        print(f"  - /{e['expected_phoneme']}/ → /{e['produced_phoneme']}/ "
              f"en position '{e['position']}'")
        populator.add_error_from_model_output(patient, e)

    # Parser et ajouter la deuxième sortie
    errors_2 = populator.parse_model_output(model_output_multi)
    print(f"\nErreurs parsées (mot 2) : {len(errors_2)}")
    for e in errors_2:
        print(f"  - /{e['expected_phoneme']}/ → /{e['produced_phoneme']}/ "
              f"en position '{e['position']}'")
        populator.add_error_from_model_output(patient, e)

    populator.print_statistics()
    populator.save_ontology(OUTPUT_PATH)
    print("Exemple 2 terminé.")


# ==============================================================================
# Exemple 3 : Traitement par lot depuis le CSV
# ==============================================================================

def exemple_3_traitement_csv():
    """
    Traiter plusieurs enregistrements depuis le fichier audio_db.csv.

    Simule ce qui se passe quand on a analysé plusieurs patients.
    """
    print("\n" + "=" * 60)
    print("EXEMPLE 3 : Traitement par lot depuis le CSV")
    print("=" * 60)

    if not os.path.exists(LABEL_CSV_PATH):
        print(f"  [INFO] Fichier CSV introuvable : {LABEL_CSV_PATH}")
        print("  Cet exemple nécessite le fichier audio_db.csv.")
        return

    populator = OntologyPopulator(ONTOLOGY_PATH)

    # Simuler des sorties du modèle pour plusieurs patients
    # En production, ces textes viendraient du vrai modèle d'analyse
    model_outputs = {
        "P02_S02_W_F01.wav": """
Reference IPA (correct):   roʃ
Produced IPA  (detected):  ros
Total phonemes in reference word: 3
Number of ʃ errors detected:      1
Error 1 of 1:
Expected phoneme:   [ʃ]
Produced phoneme:   [s]
Position in word:   final (end of word)
Preceded by:        [o]
Followed by:        [(none)]
""",
        "P02_S02_W_I01.wav": """
Reference IPA (correct):   ʃa
Produced IPA  (detected):  sa
Total phonemes in reference word: 2
Number of ʃ errors detected:      1
Error 1 of 1:
Expected phoneme:   [ʃ]
Produced phoneme:   [s]
Position in word:   initial (beginning of word)
Preceded by:        [(none)]
Followed by:        [a]
""",
    }

    # Traitement par lot
    results = process_batch_from_csv(
        populator=populator,
        label_csv_path=LABEL_CSV_PATH,
        model_outputs=model_outputs,
        verbose=True,
    )

    populator.print_statistics()
    populator.save_ontology(OUTPUT_PATH)
    print(f"\nRésultat : {results['errors_added']} erreur(s) ajoutée(s)")
    print("Exemple 3 terminé.")


# ==============================================================================
# Exemple 4 : Explorer les items phonétiques
# ==============================================================================

def exemple_4_items_phonetiques():
    """
    Charger et explorer les métadonnées des items phonétiques.

    Montre comment construire des exercices à partir du catalogue d'items.
    """
    print("\n" + "=" * 60)
    print("EXEMPLE 4 : Items phonétiques depuis le catalogue")
    print("=" * 60)

    if not os.path.exists(META_CSV_PATH):
        print(f"  [INFO] Fichier CSV introuvable : {META_CSV_PATH}")
        print("  Cet exemple nécessite le fichier exercices_a_realiser.csv.")
        return

    # Charger les métadonnées
    items = load_items_metadata(META_CSV_PATH)

    # Afficher un résumé
    print(f"\nNombre total d'items : {len(items)}")

    # Grouper par type
    types: dict = {}
    for item in items.values():
        t = item.type_item or "Autre"
        types[t] = types.get(t, 0) + 1

    print("\nItems par type :")
    for t, count in sorted(types.items()):
        print(f"  {t:15s} : {count}")

    # Grouper par position
    positions: dict = {}
    for item in items.values():
        p = item.position or "Non définie"
        positions[p] = positions.get(p, 0) + 1

    print("\nItems par position :")
    for p, count in sorted(positions.items()):
        print(f"  {p:15s} : {count}")

    # Afficher quelques exemples de mots
    print("\nExemples de mots (type=Mot) :")
    mots = [i for i in items.values() if i.type_item.lower() == "mot"][:5]
    for m in mots:
        print(f"  {m.item_id:8s} | {m.mot:15s} | {m.position:10s} | "
              f"avant={m.voyelle_precedente or '-':5s} après={m.voyelle_suivante or '-'}")

    # Construire les erreurs depuis les items (pour peupler l'ontologie)
    populator = OntologyPopulator(ONTOLOGY_PATH)
    patient_catalogue = PatientInfo(
        patient_id="CATALOGUE",
        speaker_id="CATALOGUE",
        age=0,
        sexe="?",
        collecteur_id="CATALOGUE",
    )

    errors_from_items = build_errors_from_items(
        items, phoneme_travail="ʃ", filter_type="Mot"
    )
    print(f"\nErreurs construites depuis les mots : {len(errors_from_items)}")

    for error in errors_from_items:
        populator.add_error_from_model_output(patient_catalogue, error)

    populator.print_statistics()
    populator.save_ontology(OUTPUT_PATH)
    print("Exemple 4 terminé.")


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    print("Back2speaK - Exemples d'utilisation du système ontologie")
    print("=" * 60)

    # Vérifier que l'ontologie existe
    if not os.path.exists(ONTOLOGY_PATH):
        print(f"\n[ERREUR] Fichier ontologie introuvable : {ONTOLOGY_PATH}")
        print("Placez le fichier 'ontologie.owx' dans le dossier 'ontologie/'")
        sys.exit(1)

    # Lancer les exemples (commenter/décommenter selon besoin)
    exemple_1_ajout_manuel()
    # exemple_2_parse_modele()
    # exemple_3_traitement_csv()
    # exemple_4_items_phonetiques()

    print("\nTous les exemples terminés avec succès.")
