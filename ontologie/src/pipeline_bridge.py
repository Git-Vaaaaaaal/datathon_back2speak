"""
PipelineBridge - Connexion entre features_extraction_wav2vecFr.ipynb et l'ontologie

Ce module fait le lien direct entre la sortie du notebook d'analyse phonétique
et le peuplement de l'ontologie OWL Back2speaK.

La sortie du notebook (variable `results`) contient pour chaque fichier audio :
    - audio_id, speaker, age
    - sh_errors: liste d'erreurs sur /ʃ/ avec position, phonèmes voisins, etc.
    - success_rate: taux de réussite global du fichier (0.0 → 1.0)
    - reference_ipa / produced_ipa

Ce module :
    1. Regroupe les résultats par patient
    2. Calcule le taux de réussite AGRÉGÉ par (patient, position)
       → ex: S02 a 3 fichiers en position finale → taux moyen = 33%
    3. Peuple l'ontologie avec ces données
    4. Sauvegarde sous 'ontologie_peuplée.owl'

Utilisation depuis le notebook (copier-coller la dernière cellule) :
    from ontologie.src.pipeline_bridge import populate_ontology_from_notebook
    populate_ontology_from_notebook(results, df)

Ou depuis un script standalone :
    python ontologie/run_pipeline.py
"""

import os
import sys
from collections import defaultdict
from typing import List, Dict, Optional

# Ajout du répertoire parent pour trouver le package src
_HERE = os.path.dirname(os.path.abspath(__file__))
_ONTOLOGIE_DIR = os.path.dirname(_HERE)
if _ONTOLOGIE_DIR not in sys.path:
    sys.path.insert(0, _ONTOLOGIE_DIR)

from .ontology_populator import OntologyPopulator, PatientInfo


# ==============================================================================
# Chemins par défaut
# ==============================================================================

DEFAULT_ONTOLOGY_PATH = os.path.join(_ONTOLOGIE_DIR, "ontologie.owx")
DEFAULT_OUTPUT_PATH   = os.path.join(_ONTOLOGIE_DIR, "output", "ontologie_peuplée.owl")


# ==============================================================================
# Conversion du format notebook → format ontologie
# ==============================================================================

def _build_patient_lookup(df) -> Dict[str, PatientInfo]:
    """
    Construit un dictionnaire {speaker_id: PatientInfo} depuis le DataFrame.

    Le DataFrame `df` doit avoir les colonnes :
        speaker, age, sexe, collecteur_id
    (c'est le df fusionné produit par la cellule 3 du notebook)
    """
    patients = {}

    for _, row in df.drop_duplicates(subset=["speaker"]).iterrows():
        speaker = str(row.get("speaker", "")).strip()
        if not speaker:
            continue

        collecteur = str(row.get("collecteur_id", "")).strip()
        patient_id = f"{collecteur}_{speaker}" if collecteur else speaker

        try:
            age = int(row.get("age", 0))
        except (ValueError, TypeError):
            age = 0

        patients[speaker] = PatientInfo(
            patient_id=patient_id,
            speaker_id=speaker,
            age=age,
            sexe=str(row.get("sexe", "?")).strip(),
            collecteur_id=collecteur,
        )

    return patients


def _aggregate_success_rates(results: list) -> Dict[tuple, float]:
    """
    Calcule le taux de réussite moyen par (speaker, position).

    Le notebook calcule un `success_rate` par fichier audio.
    Ici on fait la moyenne sur tous les fichiers du même patient
    à la même position dans le mot.

    Exemple :
        S02 a 5 fichiers en position "final" → taux moyen = 0.40 (40%)

    Retourne un dict {(speaker, position_label): taux_moyen}
    """
    rates_by_key: Dict[tuple, List[float]] = defaultdict(list)

    for r in results:
        speaker = r.get("speaker", "")
        # Ignorer les fichiers avec audio manquant (success_rate == None)
        raw_rate = r.get("success_rate")
        if raw_rate is None or r.get("file_missing"):
            continue
        file_rate = float(raw_rate)

        # Enregistrer le taux pour chaque position rencontrée dans ce fichier
        seen_positions = set()
        for err in r.get("sh_errors", []):
            pos = err.get("position_label", "medial")
            if (speaker, pos) not in seen_positions:
                rates_by_key[(speaker, pos)].append(file_rate)
                seen_positions.add((speaker, pos))

        # Si aucune erreur mais le fichier a été traité, enregistrer success=1.0
        # uniquement si is_correct est connu
        if not r.get("sh_errors") and r.get("is_correct", False):
            pos = r.get("position_du_sh", "").strip().lower() or "medial"
            rates_by_key[(speaker, pos)].append(1.0)

    return {k: sum(v) / len(v) for k, v in rates_by_key.items()}


def _notebook_error_to_ontology_dict(sh_error: dict, result: dict,
                                      success_rate: float) -> dict:
    """
    Convertit un sh_error du notebook en dictionnaire attendu par l'ontologie.

    Mapping des champs :
        sh_error["expected"]          → expected_phoneme  (toujours "ʃ")
        sh_error["produced"]          → produced_phoneme  (ex: "s", "(omitted)")
        sh_error["position_label"]    → position          (initial/medial/final)
        sh_error["preceding_vowel"]   → preceded_by
        sh_error["following_vowel"]   → followed_by
        result["reference_ipa"]       → reference_ipa
        result["produced_ipa"]        → produced_ipa
        success_rate (agrégé)         → success_rate
    """
    produced = sh_error.get("produced", "(omitted)")
    # Normaliser "(omitted)" → "?" pour l'ontologie
    if produced in ("(omitted)", None, ""):
        produced = "?"

    return {
        "expected_phoneme": sh_error.get("expected", "ʃ"),
        "produced_phoneme": produced,
        "position":         sh_error.get("position_label", "medial"),
        "preceded_by":      sh_error.get("preceding_vowel", "(none)"),
        "followed_by":      sh_error.get("following_vowel", "(none)"),
        "reference_ipa":    result.get("reference_ipa", ""),
        "produced_ipa":     result.get("produced_ipa", ""),
        "success_rate":     success_rate,
    }


# ==============================================================================
# Fonction principale
# ==============================================================================

def populate_ontology_from_notebook(
    results: list,
    df,
    ontology_path: str = DEFAULT_ONTOLOGY_PATH,
    output_path:   str = DEFAULT_OUTPUT_PATH,
    only_errors:   bool = True,
    verbose:       bool = True,
) -> OntologyPopulator:
    """
    Peuple l'ontologie OWL avec les résultats du notebook d'analyse phonétique.

    Cette fonction est conçue pour être appelée directement depuis le notebook
    après l'exécution du pipeline complet.

    Args:
        results       : Variable `results` produite par la cellule 6 du notebook
                        (liste de dicts retournés par analyze_audio())
        df            : DataFrame `df` produit par la cellule 3 du notebook
                        (données fusionnées audio + items, avec speaker/age/sexe)
        ontology_path : Chemin vers l'ontologie de base (ontologie.owx)
        output_path   : Chemin de sortie pour l'ontologie enrichie
        only_errors   : Si True (défaut), n'ajoute que les fichiers avec erreurs ʃ.
                        Si False, ajoute aussi les patients sans erreur (succès complet).
        verbose       : Afficher la progression

    Returns:
        L'instance OntologyPopulator après peuplement (pour inspection si besoin)

    Exemple d'appel depuis le notebook :
        # Cellule finale du notebook (après avoir obtenu `results` et `df`)

        import sys, os
        sys.path.insert(0, os.path.join(os.getcwd(), "ontologie"))

        from src.pipeline_bridge import populate_ontology_from_notebook
        populate_ontology_from_notebook(results, df)
    """
    if verbose:
        print("=" * 60)
        print("PEUPLEMENT DE L'ONTOLOGIE BACK2SPEAK")
        print("=" * 60)

    # --- Charger l'ontologie ---
    populator = OntologyPopulator(ontology_path)

    # --- Construire le dictionnaire des patients ---
    patients_lookup = _build_patient_lookup(df)
    if verbose:
        print(f"[INFO] {len(patients_lookup)} patients identifiés depuis le DataFrame")

    # --- Calculer les taux de réussite agrégés ---
    agg_rates = _aggregate_success_rates(results)
    if verbose:
        print(f"[INFO] {len(agg_rates)} combinaisons (patient × position) calculées")

    # --- Parcourir tous les résultats ---
    stats = {
        "fichiers_traites":   0,
        "fichiers_ignores":   0,
        "erreurs_ajoutees":   0,
        "patients_nouveaux":  0,
    }

    known_patients_before = set(patients_lookup.keys())

    for result in results:
        speaker    = result.get("speaker", "")
        sh_errors  = result.get("sh_errors", [])
        n_errors   = result.get("n_sh_errors", len(sh_errors))
        audio_id   = result.get("audio_id", "?")
        is_correct = result.get("is_correct", False)

        # Ignorer les fichiers avec données invalides (audio manquant)
        if result.get("file_missing") or result.get("success_rate") is None:
            stats["fichiers_ignores"] += 1
            continue

        # Filtrer les fichiers sans erreur si only_errors=True
        if only_errors and n_errors == 0:
            stats["fichiers_ignores"] += 1
            continue

        # Identifier le patient
        if speaker not in patients_lookup:
            if verbose:
                print(f"  [SKIP] Speaker inconnu : {speaker} ({audio_id})")
            stats["fichiers_ignores"] += 1
            continue

        patient_info = patients_lookup[speaker]

        # Traiter chaque erreur ʃ du fichier
        added_in_file = 0
        for sh_error in sh_errors:
            position     = sh_error.get("position_label", "medial")
            success_rate = agg_rates.get((speaker, position), 0.0)

            error_dict = _notebook_error_to_ontology_dict(
                sh_error, result, success_rate
            )

            try:
                populator.add_error_from_model_output(patient_info, error_dict)
                added_in_file += 1
            except Exception as e:
                if verbose:
                    print(f"  [ERREUR] {audio_id} : {e}")

        stats["erreurs_ajoutees"] += added_in_file
        stats["fichiers_traites"] += 1

        if verbose and added_in_file > 0:
            rate_pct = agg_rates.get((speaker, "final"), 0.0) * 100
            print(f"  [OK] {audio_id:<35s} "
                  f"{added_in_file} erreur(s) | "
                  f"taux={result.get('success_rate', 0)*100:.0f}%")

    # --- Afficher les résultats ---
    if verbose:
        print()
        print("-" * 60)
        print(f"Fichiers avec erreurs traités : {stats['fichiers_traites']}")
        print(f"Fichiers ignorés (corrects)   : {stats['fichiers_ignores']}")
        print(f"Erreurs ʃ ajoutées            : {stats['erreurs_ajoutees']}")
        print("-" * 60)
        populator.print_statistics()

    # --- Sauvegarder ---
    populator.save_ontology(output_path)

    if verbose:
        print()
        print(f"[OK] Ontologie peuplée sauvegardée :")
        print(f"     {os.path.abspath(output_path)}")
        print()
        print("Vous pouvez maintenant ouvrir ce fichier dans Protégé.")
        print("=" * 60)

    return populator


# ==============================================================================
# Fonction utilitaire : export JSON des résultats (intermédiaire)
# ==============================================================================

def export_results_to_json(results: list, output_path: str) -> None:
    """
    Exporte les résultats du notebook dans un fichier JSON.

    Utile pour sauvegarder les résultats et les réutiliser plus tard
    sans avoir à relancer le notebook (qui prend 5-10 minutes).

    Args:
        results     : Variable `results` du notebook
        output_path : Chemin du fichier JSON de sortie

    Exemple :
        export_results_to_json(results, "ontologie/output/resultats_pipeline.json")
    """
    import json

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Sérialiser les résultats (supprimer les champs non-JSON-sérialisables)
    serializable = []
    for r in results:
        entry = {
            "audio_id":      r.get("audio_id", ""),
            "speaker":       r.get("speaker", ""),
            "age":           r.get("age", 0),
            "target_word":   r.get("target_word", ""),
            "label":         r.get("label", ""),
            "reference_ipa": r.get("reference_ipa", ""),
            "produced_ipa":  r.get("produced_ipa", ""),
            "n_sh_errors":   r.get("n_sh_errors", 0),
            "success_rate":  r.get("success_rate", 0.0),
            "per":           r.get("per", 0.0),
            "is_correct":    r.get("is_correct", False),
            "position_du_sh": r.get("position_du_sh", ""),
            "sh_errors": [
                {
                    "expected":         e.get("expected", "ʃ"),
                    "produced":         e.get("produced", "?"),
                    "position_label":   e.get("position_label", ""),
                    "preceding_vowel":  e.get("preceding_vowel", "(none)"),
                    "following_vowel":  e.get("following_vowel", "(none)"),
                    "error_type":       e.get("error_type", ""),
                    "target_word":      e.get("target_word", ""),
                }
                for e in r.get("sh_errors", [])
            ],
        }
        serializable.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(serializable)} résultats exportés → {output_path}")


def populate_ontology_from_json(
    json_path: str,
    df,
    ontology_path: str = DEFAULT_ONTOLOGY_PATH,
    output_path:   str = DEFAULT_OUTPUT_PATH,
    verbose: bool = True,
) -> OntologyPopulator:
    """
    Peuple l'ontologie depuis un fichier JSON (exporté par export_results_to_json).

    Utile pour relancer le peuplement sans relancer le notebook.

    Args:
        json_path     : Chemin vers le fichier JSON des résultats
        df            : DataFrame `df` du notebook (pour les infos patients)
        ontology_path : Chemin vers l'ontologie de base
        output_path   : Chemin de sortie
        verbose       : Afficher la progression
    """
    import json

    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"[OK] {len(results)} résultats chargés depuis {json_path}")
    return populate_ontology_from_notebook(
        results, df,
        ontology_path=ontology_path,
        output_path=output_path,
        verbose=verbose,
    )
