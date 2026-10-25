"""Part 2, Task D: train a model on the ETL output and save it.

Command line:
    python -m quakes.train --task regression --model linear
    python -m quakes.train --task regression --model poly --degree 3 --alpha 10
    python -m quakes.train --task classification --model logistic [--balanced]

Steps (see the lab handout):
 1. load the table from the SQLite database written by quakes.etl
 2. X = everything except 'mag' and 'big_quake'; y = 'mag' or 'big_quake'
 3. train/test split (stratified for classification, seeded with config.SEED)
 4. fit the dummy baseline AND your model; report both on the test set
 5. cross-validate on the TRAIN set only
 6. classification only: pick the threshold from out-of-fold training
    predictions (cross_val_predict, method='predict_proba'), then apply it to test
 7. save the pipeline to models/ and the report (JSON) to reports/
"""
import argparse
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from . import config
from .etl import DB_PATH
from .evaluate import (best_threshold, classification_report_dict,
                       cross_validate_model, regression_report)
from .models import make_classifier, make_regressor

MODELS_DIR = config.ROOT / "models"
REPORTS_DIR = config.ROOT / "reports"
TARGETS = {"regression": "mag", "classification": config.TARGET}


def load_data(db_path=None):
    db_path = db_path or DB_PATH
    if not db_path.exists():
        raise SystemExit(f"{db_path} not found: run `python -m quakes.etl` first")
    with sqlite3.connect(db_path) as con:
        return pd.read_sql("select * from quakes", con)


def _print_side_by_side(reports: dict) -> None:
    table = pd.DataFrame(reports).round(4)
    print(table.to_string())


def _regression(args, X_tr, X_te, y_tr, y_te) -> tuple:
    baseline = make_regressor("dummy").fit(X_tr, y_tr)
    model = make_regressor(args.model, degree=args.degree, alpha=args.alpha).fit(X_tr, y_tr)
    report = {
        "test": {"baseline": regression_report(y_te, baseline.predict(X_te)),
                 "model": regression_report(y_te, model.predict(X_te))},
        "train": {"model": regression_report(y_tr, model.predict(X_tr))},
        "cv_train_r2": cross_validate_model(
            make_regressor(args.model, degree=args.degree, alpha=args.alpha),
            X_tr, y_tr, scoring="r2"),
    }
    print("\nTest set (baseline vs model):")
    _print_side_by_side(report["test"])
    print(f"\nTrain R2 {report['train']['model']['r2']:.4f}   "
          f"CV R2 on train {report['cv_train_r2']['mean']:.4f} +/- {report['cv_train_r2']['std']:.4f}")
    return model, report


def _classification(args, X_tr, X_te, y_tr, y_te) -> tuple:
    class_weight = "balanced" if args.balanced else None
    make = lambda: make_classifier(args.model, C=args.C, class_weight=class_weight)
    baseline = make_classifier("dummy").fit(X_tr, y_tr)
    model = make().fit(X_tr, y_tr)

    # threshold from out-of-fold TRAIN probabilities; the test set is touched once
    oof = cross_val_predict(make(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
    threshold = best_threshold(y_tr, oof, metric=args.metric)

    base_p = baseline.predict_proba(X_te)[:, 1]
    p = model.predict_proba(X_te)[:, 1]
    report = {
        "threshold": threshold,
        "threshold_metric": args.metric,
        "test": {
            "baseline": classification_report_dict(y_te, (base_p >= 0.5).astype(int), base_p),
            "model@0.5": classification_report_dict(y_te, (p >= 0.5).astype(int), p),
            f"model@{threshold}": classification_report_dict(y_te, (p >= threshold).astype(int), p),
        },
        "train": {"model@0.5": classification_report_dict(
            y_tr, model.predict(X_tr), model.predict_proba(X_tr)[:, 1])},
        "cv_train_roc_auc": cross_validate_model(make(), X_tr, y_tr, scoring="roc_auc"),
        "positive_rate": {"train": float(y_tr.mean()), "test": float(y_te.mean())},
    }
    print("\nTest set (baseline vs model):")
    _print_side_by_side(report["test"])
    print(f"\nThreshold chosen on out-of-fold train predictions ({args.metric}): {threshold}")
    print(f"CV ROC-AUC on train {report['cv_train_roc_auc']['mean']:.4f} "
          f"+/- {report['cv_train_roc_auc']['std']:.4f}")
    return model, report


def main() -> None:
    p = argparse.ArgumentParser(description="Train a model on the quakes SQLite table.")
    p.add_argument("--task", choices=list(TARGETS), required=True)
    p.add_argument("--model", required=True,
                   help="regression: dummy, linear, ridge, poly; classification: dummy, logistic")
    p.add_argument("--degree", type=int, default=2, help="polynomial degree (poly only)")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge strength (ridge, poly)")
    p.add_argument("--C", type=float, default=1.0, help="inverse regularisation (logistic)")
    p.add_argument("--balanced", action="store_true", help="class_weight='balanced' (logistic)")
    p.add_argument("--metric", default="f1", choices=["f1", "precision", "recall"],
                   help="metric for the threshold search (classification)")
    p.add_argument("--db", default=None, help="SQLite path (default: data/processed/quakes.db)")
    args = p.parse_args()

    from pathlib import Path
    df = load_data(Path(args.db) if args.db else None)
    target = TARGETS[args.task]
    X = df.drop(columns=["mag", config.TARGET, "id"], errors="ignore")
    y = df[target]
    stratify = y if args.task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=stratify,
                                              random_state=config.SEED)
    print(f"{len(df)} rows from the database; train {X_tr.shape}, test {X_te.shape}; "
          f"task={args.task}, target={target}, model={args.model}")

    run = _regression if args.task == "regression" else _classification
    model, report = run(args, X_tr, X_te, y_tr, y_te)

    name = args.model + (f"_deg{args.degree}" if args.model == "poly" else "") \
        + ("_balanced" if args.balanced else "")
    report = {"task": args.task, "model": args.model, "params": vars(args),
              "rows": len(df), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **report}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{args.task}_{name}.joblib"
    report_path = REPORTS_DIR / f"{args.task}_{name}.json"
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {model_path.relative_to(config.ROOT)} and {report_path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
