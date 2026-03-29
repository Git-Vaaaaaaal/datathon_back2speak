"""
run_pipeline.py — Script standalone Back2speaK

Ce script fait le lien entre la sortie sauvegardée du notebook
features_extraction_wav2vecFr.ipynb et l'ontologie OWL.

Deux façons de l'utiliser :

─── Option A : Appel depuis le NOTEBOOK (recommandé) ────────────────────────

    Ajoutez une cellule à la fin de features_extraction_wav2vecFr.ipynb :

        # ─── Cellule finale : peuplement de l'ontologie ──────────────────
        import sys, os
        # Ajuster ce chemin si le notebook n'est pas à la racine du projet
        sys.path.insert(0, os.getcwd())

        from ontologie.src.pipeline_bridge import (
            populate_ontology_from_notebook,
            export_results_to_json,
        )

        # (Optionnel) Sauvegarder les résultats en JSON pour usage futur
        export_results_to_json(results, "ontologie/output/resultats_pipeline.json")

        # Peupler et sauvegarder l'ontologie
        populate_ontology_from_notebook(
            results  = results,   # variable produite par la cellule 6
            df       = df,        # variable produite par la cellule 3
            output_path = "ontologie/output/ontologie_peuplée.owl"
        )

─── Option B : Script autonome (si résultats JSON déjà sauvegardés) ─────────

    1. D'abord, exporter depuis le notebook :
           export_results_to_json(results, "ontologie/output/resultats_pipeline.json")

    2. Puis lancer ce script :
           python ontologie/run_pipeline.py

─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import csv

# ── Chemins ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)

ONTOLOGY_PATH = os.path.join(SCRIPT_DIR, "ontologie.owx")
OUTPUT_PATH   = os.path.join(SCRIPT_DIR, "output", "ontologie_peuplée.owl")
JSON_RESULTS  = os.path.join(SCRIPT_DIR, "output", "resultats_pipeline.json")
AUDIO_CSV     = os.path.join(PROJECT_ROOT, "Donnees", "ch", "audio_db.csv")
ITEMS_CSV     = os.path.join(PROJECT_ROOT, "Donnees", "ch", "exercices_a_realiser.csv")

sys.path.insert(0, SCRIPT_DIR)

from src.pipeline_bridge import populate_ontology_from_notebook


# ── Chargement du CSV comme substitut au DataFrame pandas ────────────────────

class _SimpleDataFrame:
    """
    Substitut léger à pandas.DataFrame pour ce script standalone.
    Charge le CSV en mémoire et simule drop_duplicates + iterrows.
    """
    def __init__(self, rows: list):
        self._rows = rows

    def drop_duplicates(self, subset=None):
        seen = set()
        unique = []
        for row in self._rows:
            key = tuple(row.get(k, "") for k in (subset or []))
            if key not in seen:
                seen.add(key)
                unique.append(row)
        return _SimpleDataFrame(unique)

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, _SimpleRow(row)


class _SimpleRow:
    def __init__(self, data: dict):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def _load_csv_as_df(csv_path: str) -> _SimpleDataFrame:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    return _SimpleDataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Back2speaK — Peuplement de l'ontologie (script standalone)")
    print("=" * 60)
    print()

    # Vérifier que le JSON des résultats existe
    if not os.path.exists(JSON_RESULTS):
        print(f"[ERREUR] Fichier de résultats introuvable :")
        print(f"         {JSON_RESULTS}")
        print()
        print("Pour générer ce fichier, ajoutez dans le notebook :")
        print()
        print("    from ontologie.src.pipeline_bridge import export_results_to_json")
        print("    export_results_to_json(results, 'ontologie/output/resultats_pipeline.json')")
        print()
        sys.exit(1)

    # Charger les résultats JSON
    print(f"Chargement des résultats : {JSON_RESULTS}")
    with open(JSON_RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    print(f"  → {len(results)} fichiers audio chargés")

    # Charger le CSV patients comme DataFrame léger
    if not os.path.exists(AUDIO_CSV):
        print(f"[ERREUR] CSV patients introuvable : {AUDIO_CSV}")
        sys.exit(1)

    print(f"Chargement du CSV patients : {AUDIO_CSV}")
    df = _load_csv_as_df(AUDIO_CSV)

    # Peupler l'ontologie
    print()
    populate_ontology_from_notebook(
        results=results,
        df=df,
        ontology_path=ONTOLOGY_PATH,
        output_path=OUTPUT_PATH,
        verbose=True,
    )


if __name__ == "__main__":
    main()
