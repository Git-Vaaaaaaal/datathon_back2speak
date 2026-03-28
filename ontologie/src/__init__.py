"""
Package Back2speaK Ontologie
Peuplement automatique de l'ontologie OWL à partir des erreurs phonétiques.
"""
from .ontology_populator import OntologyPopulator, PatientInfo
from .data_processor import load_patients_from_csv, load_items_metadata, process_batch_from_csv

__all__ = [
    "OntologyPopulator",
    "PatientInfo",
    "load_patients_from_csv",
    "load_items_metadata",
    "process_batch_from_csv",
]
