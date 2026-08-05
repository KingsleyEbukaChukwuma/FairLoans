import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from configs.config import (
    COMPARISON_DIR,
    CV_FOLDS,
    DATASET_FILENAME,
    DEPENDENCE_FEATURES,
    EXECUTIVE_SUMMARY_FILENAME,
    EXPLAINABILITY_DIR,
    FAIRNESS_COMPARISON_FILENAME,
    FAIRNESS_CONSTRAINT,
    FAIRNESS_DIR,
    FAIRNESS_MITIGATION_METHOD,
    FAIRNESS_OBJECTIVE,
    FAIRNESS_SUMMARY_FILENAME,
    FEATURE_NAMES_FILENAME,
    GOVERNANCE_REPORT_FILENAME,
    INPUT_SCHEMA_FILENAME,
    METRICS_FILENAME,
    MODEL_DIR,
    OPTIMIZATION_METRIC,
    OPTUNA_TRIALS,
    PIPELINE_FILENAME,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORT_DIR,
    RESULTS_FILENAME,
    SHAP_EXPLAINER_FILENAME,
    SHAP_INTERACTION_VALUES_FILENAME,
    SHAP_VALUES_FILENAME,
    STUDY_DIR,
    TEST_SIZE,
    THRESHOLD_FILENAME,
    THRESHOLD_OPTIMIZATION_METRIC,
    TRAINING_LOG,
    TRAINING_SUMMARY_FILENAME,
)
from src.data.loader import DataLoader
from src.explainability.beeswarm import BeeswarmPlot
from src.explainability.dependence import DependencePlot
from src.explainability.explainer import SHAPExplainer
from src.explainability.global_importance import GlobalFeatureImportance
from src.explainability.interactions import InteractionPlot
from src.explainability.local_explanation import LocalExplanation
from src.explainability.waterfall import WaterfallPlot
from src.fairness.comparison import FairnessComparison
from src.fairness.evaluator import FairnessEvaluator
from src.fairness.governance import GovernanceReport
from src.fairness.mitigation import BiasMitigator
from src.features.preprocessing import CreditPreprocessor
from src.models.build_pipeline import build_pipeline
from src.models.evaluate import ModelEvaluator
from src.models.evaluation_plots import EvaluationPlots
from src.models.registry import MODELS
from src.models.save import (
    save_dataframe,
    save_json,
    save_model,
    save_pickle,
    save_study,
)
from src.models.select_best import select_best
from src.models.threshold import DecisionThresholdOptimizer
from src.models.trainer import ModelTrainer
from src.models.tune import ModelTuner
from src.utils.logger import get_logger


class TrainingManager:

    def __init__(self):

        self.loader = DataLoader()
        self.preprocessor = CreditPreprocessor()

        self.results = []
        self.trained_models = {}

        self.sensitive_train = None
        self.sensitive_test = None
        self.logger = get_logger(
            "training",
            TRAINING_LOG,
        )

    def load_data(self):

        return self.loader.load_arff(RAW_DATA_DIR / DATASET_FILENAME)

    def split_data(self, df):

        X, y, sensitive = self.preprocessor.prepare_data(df)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_train,
            sensitive_test,
        ) = train_test_split(
            X,
            y,
            sensitive,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_train,
            sensitive_test,
        )

    def train_models(self, X_train, X_test, y_train, y_test):

        transformer = self.preprocessor.build_transformer()

        for model_name in MODELS:

            self.logger.info("=" * 60)

            self.logger.info(f"Training model: {model_name}")

            self.logger.info("=" * 60)

            pipeline = build_pipeline(
                transformer,
                model_name,
            )

            tuner = ModelTuner(
                pipeline,
                X_train,
                y_train,
            )

            study = tuner.optimize(
                model_name=model_name,
                n_trials=OPTUNA_TRIALS,
            )

            save_study(
                study,
                STUDY_DIR / f"{model_name}_study.joblib",
            )

            save_json(
                study.best_params,
                STUDY_DIR / f"{model_name}_best_params.json",
            )
            pipeline.named_steps["classifier"].set_params(**study.best_params)

            trainer = ModelTrainer(pipeline)

            trainer.fit(
                X_train,
                y_train,
            )

            metrics = ModelEvaluator.evaluate(
                trainer.pipeline,
                X_test,
                y_test,
            )

            metrics["Model"] = model_name

            self.results.append(metrics)

            self.trained_models[model_name] = trainer

            self.logger.info(metrics)

    def save_best_model(self, X_train, X_test, y_train, y_test, dataset_size):

        best = select_best(self.results)

        best_name = best["Model"]

        save_json(
            best.to_dict(),
            MODEL_DIR / METRICS_FILENAME,
        )

        self.logger.info(f"Best model selected: {best_name}")

        trainer = self.trained_models[best_name]

        trainer.calibrate(
            X_train,
            y_train,
        )

        y_prob = trainer.calibrated_pipeline.predict_proba(
            X_test,
        )[:, 1]

        y_pred = trainer.calibrated_pipeline.predict(
            X_test,
        )

        EvaluationPlots.build(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        best_threshold, best_score = DecisionThresholdOptimizer.optimize(
            y_true=y_test,
            y_prob=y_prob,
            metric=THRESHOLD_OPTIMIZATION_METRIC,
        )

        save_json(
            {
                "threshold": float(best_threshold),
                "metric": "balanced_accuracy",
                "score": float(best_score),
            },
            MODEL_DIR / THRESHOLD_FILENAME,
        )
        save_model(
            trainer.calibrated_pipeline,
            MODEL_DIR / PIPELINE_FILENAME,
        )

        summary = {
            "best_model": best["Model"],
            "models_trained": len(self.results),
            "optimization_metric": OPTIMIZATION_METRIC,
            "calibrated": True,
            "cross_validation_folds": CV_FOLDS,
            "optuna_trials": OPTUNA_TRIALS,
            "dataset_size": dataset_size,
        }

        save_json(
            summary,
            MODEL_DIR / TRAINING_SUMMARY_FILENAME,
        )

        return trainer

    def responsible_ai_pipeline(
        self,
        trainer,
        X_train,
        X_test,
        y_train,
        y_test,
    ):
        """
        Execute the Responsible AI workflow.

        1. Evaluate baseline fairness
        2. Apply bias mitigation
        3. Evaluate mitigated model
        4. Compare baseline vs mitigated
        5. Save all artifacts
        """

        self.logger.info("Evaluating baseline fairness...")

        # ---------------------------------------------------------
        # Baseline predictions
        # ---------------------------------------------------------

        baseline_pred = trainer.calibrated_pipeline.predict(X_test)

        baseline_prob = trainer.calibrated_pipeline.predict_proba(X_test)[:, 1]

        baseline_performance = ModelEvaluator.evaluate(
            trainer.calibrated_pipeline,
            X_test,
            y_test,
        )

        save_json(
            baseline_performance,
            FAIRNESS_DIR / "baseline_performance.json",
        )

        baseline_fairness = FairnessEvaluator.evaluate(
            y_true=y_test,
            y_pred=baseline_pred,
            sensitive_features=self.sensitive_test,
        )

        FairnessEvaluator.save(
            baseline_fairness,
            stage="baseline",
        )

        # ---------------------------------------------------------
        # Bias Mitigation
        # ---------------------------------------------------------

        self.logger.info(f"Applying {FAIRNESS_MITIGATION_METHOD}...")
        if FAIRNESS_MITIGATION_METHOD == "threshold_optimizer":

            mitigator = BiasMitigator.threshold_optimizer(
                estimator=trainer.calibrated_pipeline,
                X_train=X_train,
                y_train=y_train,
                sensitive_features=self.sensitive_train,
                constraints=FAIRNESS_CONSTRAINT,
                objective=FAIRNESS_OBJECTIVE,
            )

        elif FAIRNESS_MITIGATION_METHOD == "exponentiated_gradient":

            mitigator = BiasMitigator.exponentiated_gradient(
                estimator=trainer.pipeline,
                X_train=X_train,
                y_train=y_train,
                sensitive_features=self.sensitive_train,
                constraint=FAIRNESS_CONSTRAINT,
            )

        else:
            raise ValueError(f"Unknown fairness method: {FAIRNESS_MITIGATION_METHOD}")

        mitigated_pred = BiasMitigator.predict(
            mitigator,
            X_test,
            self.sensitive_test,
        )

        if FAIRNESS_MITIGATION_METHOD == "threshold_optimizer":

            mitigated_prob = baseline_prob

            mitigated_performance = ModelEvaluator.classification_metrics(
                y_test,
                mitigated_pred,
                mitigated_prob,
            )

            mitigated_performance.update(
                ModelEvaluator.credit_metrics(
                    y_test,
                    mitigated_prob,
                )
            )

            mitigated_performance["Confusion Matrix"] = confusion_matrix(
                y_test,
                mitigated_pred,
            ).tolist()

        elif FAIRNESS_MITIGATION_METHOD == "exponentiated_gradient":

            mitigated_performance = ModelEvaluator.basic_evaluation(
                y_test,
                mitigated_pred,
            )

            mitigated_performance.update(
                {
                    "ROC AUC": None,
                    "PR AUC": None,
                    "Log Loss": None,
                    "Brier Score": None,
                    "Gini": None,
                    "KS": None,
                }
            )

            mitigated_performance["Confusion Matrix"] = confusion_matrix(
                y_test,
                mitigated_pred,
            ).tolist()

        mitigated_fairness = FairnessEvaluator.evaluate(
            y_true=y_test,
            y_pred=mitigated_pred,
            sensitive_features=self.sensitive_test,
        )

        save_json(
            mitigated_performance,
            FAIRNESS_DIR / "mitigated_performance.json",
        )

        FairnessEvaluator.save(
            mitigated_fairness,
            stage="mitigated",
        )

        # ---------------------------------------------------------
        # Comparison
        # ---------------------------------------------------------

        comparison = FairnessComparison.compare(
            baseline_performance=baseline_performance,
            mitigated_performance=mitigated_performance,
            baseline_fairness=baseline_fairness["overall"],
            mitigated_fairness=mitigated_fairness["overall"],
        )

        summary = FairnessComparison.summary(comparison)

        governance = GovernanceReport.generate(
            comparison,
        )

        executive_summary = GovernanceReport.executive_summary(
            comparison,
        )
        save_dataframe(
            comparison,
            FAIRNESS_DIR / FAIRNESS_COMPARISON_FILENAME,
        )

        save_json(
            summary,
            FAIRNESS_DIR / FAIRNESS_SUMMARY_FILENAME,
        )

        save_dataframe(
            governance,
            COMPARISON_DIR / GOVERNANCE_REPORT_FILENAME,
        )

        save_json(
            executive_summary,
            COMPARISON_DIR / EXECUTIVE_SUMMARY_FILENAME,
        )

        self.logger.info("Responsible AI pipeline completed.")

        return comparison

    def build_explainability(
        self,
        trainer,
        X_train,
        X_test,
    ):
        """
        Build and cache SHAP artifacts.

        SHAP values are expensive to compute,
        so compute them once after training
        and reuse them everywhere.
        """

        self.logger.info("Building SHAP explainability...")

        explainer = SHAPExplainer(
            trainer.pipeline,
        ).fit(
            X_train,
        )

        shap_values = explainer.shap_values(
            X_test,
        )

        interaction_values = explainer.interaction_values(
            X_test,
        )

        save_pickle(
            explainer,
            EXPLAINABILITY_DIR / SHAP_EXPLAINER_FILENAME,
        )

        save_pickle(
            shap_values,
            EXPLAINABILITY_DIR / SHAP_VALUES_FILENAME,
        )

        if interaction_values is not None:

            save_pickle(
                interaction_values,
                EXPLAINABILITY_DIR / SHAP_INTERACTION_VALUES_FILENAME,
            )

            InteractionPlot.build(
                explainer,
                interaction_values,
            )
        else:

            self.logger.warning(
                "Selected model does not support SHAP interaction values."
            )

        GlobalFeatureImportance.build(
            explainer,
            shap_values,
        )

        BeeswarmPlot.build(
            shap_values,
        )

        WaterfallPlot.build(
            shap_values,
            instance=0,
        )

        LocalExplanation.build(
            explainer,
            shap_values,
            instance=0,
        )

        for feature in DEPENDENCE_FEATURES:

            DependencePlot.build(
                shap_values=shap_values,
                feature=feature,
            )

        self.logger.info("Explainability artifacts created.")

        return explainer, shap_values

    def save_results(self):

        save_dataframe(
            pd.DataFrame(self.results),
            REPORT_DIR / RESULTS_FILENAME,
        )

    def run(self):

        self.logger.info("Loading data")

        df = self.load_data()

        self.logger.info("Splitting dataset")

        (
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_train,
            sensitive_test,
        ) = self.split_data(df)

        save_pickle(
            X_train.columns.tolist(),
            MODEL_DIR / FEATURE_NAMES_FILENAME,
        )

        self.sensitive_train = sensitive_train
        self.sensitive_test = sensitive_test

        self.logger.info("Training models")

        self.train_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        self.logger.info("Selecting best model")

        trainer = self.save_best_model(
            X_train,
            X_test,
            y_train,
            y_test,
            dataset_size=len(df),
        )

        self.logger.info("Input schema saved.")

        schema = CreditPreprocessor.get_input_schema(
            trainer.pipeline.named_steps["preprocessor"]
        )

        save_json(
            schema,
            MODEL_DIR / INPUT_SCHEMA_FILENAME,
        )

        self.logger.info("Running Responsible AI pipeline")

        self.responsible_ai_pipeline(
            trainer,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        self.logger.info("Building Explainability")

        self.build_explainability(
            trainer,
            X_train,
            X_test,
        )

        self.logger.info("Saving benchmark results")

        self.save_results()

        self.logger.info("Training completed successfully.")


if __name__ == "__main__":

    manager = TrainingManager()

    manager.run()
