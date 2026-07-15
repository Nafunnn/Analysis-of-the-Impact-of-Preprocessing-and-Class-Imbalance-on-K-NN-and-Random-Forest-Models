"""Resolusi path relatif ke root paket eksperimen-klasifikasi-depresi."""

from pathlib import Path

# app/utils/paths.py -> parents[0]=utils, [1]=app, [2]=eksperimen-klasifikasi-depresi
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PACKAGE_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
FIGURES_DIR = PACKAGE_ROOT / "results" / "figures"
TABLES_DIR = PACKAGE_ROOT / "results" / "tables"

RAW_CSV = DATA_DIR / "Teen_Mental_Health_Dataset.csv"
BEST_MODEL = ARTIFACTS_DIR / "best_fs_model.joblib"
STANDARD_SCALER = ARTIFACTS_DIR / "standard_scaler.joblib"
PREPROCESSING_META = ARTIFACTS_DIR / "preprocessing_metadata.json"
EXPERIMENT_RESULTS = ARTIFACTS_DIR / "experiment_fs_results.json"
SHAP_RESULTS = ARTIFACTS_DIR / "shap_analysis_results.json"
TEST_METRICS_CSV = TABLES_DIR / "12_test_results_fs_comparison.csv"
