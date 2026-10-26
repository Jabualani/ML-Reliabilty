from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ml_reliability import MetaPipelineClassifier, ReliablePipeline


def show_result(name: str, pipeline: ReliablePipeline, X: pd.DataFrame) -> None:
    predictions, report = pipeline.predict_with_report(X)
    print(f"\n{name}")
    preview = predictions[:8]
    suffix = " ..." if len(predictions) > len(preview) else ""
    print(f"  predictions: {preview}{suffix} ({len(predictions)} total)")
    print(f"  distribution similarity: {report.similarity_score:.3f}")
    print(f"  mean PSI: {report.mean_psi:.3f}")
    print(
        f"  similarity CI: {report.confidence_intervals['similarity_score']}; "
        f"mean PSI CI: {report.confidence_intervals['mean_psi']}"
    )
    print(f"  covariate shift detected: {report.covariate_shift_detected}")
    print(f"  drifted features: {list(report.drifted_features)}")
    for feature in report.features:
        print(
            f"    {feature.name}: "
            f"KS={feature.ks_statistic if feature.ks_statistic is not None else 'N/A'}, "
            f"KL={feature.kl_divergence:.3f}, PSI={feature.psi:.3f}, "
            f"DSI={feature.dsi if feature.dsi is not None else 'N/A'}, "
            f"chi2 p={feature.chi_square_p_value if feature.chi_square_p_value is not None else 'N/A'}, "
            f"Wasserstein={feature.wasserstein_distance if feature.wasserstein_distance is not None else 'N/A'}, "
            f"drifted={feature.drifted}"
        )
        print(f"      metric CIs: {feature.confidence_intervals}")


def main() -> None:
    X_train = pd.DataFrame(
        [
            {
                "risk_score": score,
                "region_north": float(region == "north"),
                "region_south": float(region == "south"),
                "region_west": float(region == "west"),
                "region_east": 0.0,
            }
            for score, region in [
                (0.05, "north"),
                (0.12, "south"),
                (0.18, "north"),
                (0.24, "west"),
                (0.30, "south"),
                (0.36, "north"),
                (0.42, "west"),
                (0.48, "south"),
                (0.55, "north"),
                (0.60, "west"),
                (0.66, "south"),
                (0.72, "north"),
                (0.78, "west"),
                (0.84, "south"),
                (0.90, "north"),
                (0.96, "west"),
            ]
        ]
    )
    y_train = [0] * 8 + [1] * 8

    meta_model = MetaPipelineClassifier(
        base_estimators=[
            ("logistic", LogisticRegression(max_iter=500)),
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ],
        n_folds=2,
        random_state=42,
    )
    model = ReliablePipeline(
        estimator=meta_model,
        drift_threshold=0.2,
        plot_dir="reliability_plots",
    ).fit(X_train, y_train)

    print("Fitted ReliablePipeline with ReliableML's MetaPipelineClassifier")
    print(f"  training rows: {len(X_train)}")
    print(f"  meta-classifier: {type(model.estimator.meta_classifier_).__name__}")
    print(f"  saved training plot: {model.plot_dir / 'training-distributions.png'}")

    similar_data = X_train.copy()
    shifted_data = pd.DataFrame(
        [
            {
                "risk_score": score,
                "region_north": 0.0,
                "region_south": 0.0,
                "region_west": 0.0,
                "region_east": 1.0,
            }
            for score in (1.20, 1.30, 1.40, 1.50)
        ],
        columns=X_train.columns,
    )

    show_result("Incoming data similar to training", model, similar_data)
    print(f"  saved comparison: {model.last_plot_path}")
    print(f"  saved report: {model.last_report_path}")
    show_result("Incoming data with distribution shift", model, shifted_data)
    print(f"  saved comparison: {model.last_plot_path}")
    print(f"  saved report: {model.last_report_path}")

    concept_report = model.assess_concept_shift(
        X_train, [1 - label for label in y_train]
    )
    print("\nLabeled data with a changed feature-target relationship")
    print(f"  mean conditional JS divergence: {concept_report.mean_js_divergence:.3f}")
    print(f"  shifted features: {list(concept_report.shifted_features)}")
    for feature in concept_report.features:
        print(
            f"    {feature.name}: JS={feature.js_divergence}, "
            f"KL={feature.kl_divergence}, PSI={feature.psi}"
        )

    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "reliable-model.pkl"
        model.save(artifact)
        restored_model = ReliablePipeline.load(artifact)
        restored_predictions = restored_model.predict(similar_data)
        print("\nSaved and reloaded model")
        print(f"  artifact: {artifact.name} (temporary directory)")
        print(
            f"  predictions after reload: {restored_predictions[:8]} ... "
            f"({len(restored_predictions)} total)"
        )


if __name__ == "__main__":
    main()
