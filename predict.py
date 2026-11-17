"""
Predicts new data with the model saved by `main.py` (no retraining):

    python predict.py                                   # data.predict_path -> output.predictions
    python predict.py new.csv my_predictions.csv [config.yaml]

new CSV -> the same cleaning as training (no rows dropped) -> features -> the saved pipeline
-> one row per input row: id, probability, prediction.
"""
import argparse

import joblib
import yaml

from src.predict import predict_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", nargs="?", help="CSV with the training columns (no target needed)")
    parser.add_argument("output", nargs="?", help="where to write the predictions")
    parser.add_argument("config", nargs="?", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config["output"]["model_path"])
    path = predict_new(pipeline, config, args.input, args.output)
    print(f"Predictions -> {path}")
