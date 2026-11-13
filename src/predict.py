"""Predicting unlabeled rows (e.g. adult_new.csv) with the refit pipeline."""
import os

import pandas as pd

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target


def predict_new(pipeline, config: dict, threshold: float = 0.5) -> str:
    """
    Loads `data.predict_path`, applies the same cleaning and feature split as training (but no
    de-duplication: every row needs a prediction), and saves id, probability and 0/1
    prediction to `output.predictions`. Returns the path. Rows out = rows in.
    """
    raw = load_data(config["data"]["predict_path"])
    clean = clean_dataset(raw, config["diagnostics"])
    prep = config["preprocessing"]
    X, _, _ = split_features_target(clean, config["data"], prep.get("mnar_indicator_sources", []),
                                    prep.get("feature_engineering", "none"))

    proba = pipeline.predict_proba(X)[:, 1]
    target = config["data"]["target"]
    id_column = config["diagnostics"]["id_column"]
    out = pd.DataFrame({id_column: clean[id_column].values,
                        f"proba_{target}": proba,
                        f"pred_{target}": (proba >= threshold).astype(int)})
    if len(out) != len(raw):
        raise RuntimeError(f"{len(raw)} rows in, {len(out)} predictions out")

    path = config["output"]["predictions"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return path
