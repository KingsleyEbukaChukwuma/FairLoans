import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import (
    RAW_DATA_DIR,
    REPORT_DIR,
    MODEL_DIR,
    STUDY_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    DATASET_FILENAME,
    OPTUNA_TRIALS,
    PIPELINE_FILENAME,
    RESULTS_FILENAME,
    METRICS_FILENAME,
    TRAINING_SUMMARY_FILENAME,
    FEATURE_NAMES_FILENAME,
    OPTIMIZATION_METRIC,
    CV_FOLDS,
    TRAINING_LOG,
    FAIRNESS_DIR,
    FAIRNESS_COMPARISON_FILENAME,
    FAIRNESS_SUMMARY_FILENAME,
    COMPARISON_DIR,
    GOVERNANCE_REPORT_FILENAME,
    EXECUTIVE_SUMMARY_FILENAME,
    FAIRNESS_CONSTRAINT,
    FAIRNESS_OBJECTIVE,
)

from src.fairness.evaluator import FairnessEvaluator
from src.data.loader import DataLoader
from src.features.preprocessing import CreditPreprocessor

from src.models.registry import MODELS
from src.models.build_pipeline import build_pipeline
from src.models.trainer import ModelTrainer
from src.models.tune import ModelTuner
from src.models.evaluate import ModelEvaluator
from src.models.select_best import select_best
from src.models.save import (save_model, save_study, save_json, save_dataframe, save_pickle,)
from src.utils.logger import get_logger
from src.fairness.mitigation import BiasMitigator
from src.fairness.comparison import FairnessComparison
from src.fairness.governance import GovernanceReport


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

        return self.loader.load_arff(
            RAW_DATA_DIR / DATASET_FILENAME
        )

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

            self.logger.info(
                "=" * 60
            )

            self.logger.info(
                f"Training model: {model_name}"
            )

            self.logger.info(
                "=" * 60
            )

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
            pipeline.named_steps["classifier"].set_params(
                **study.best_params
            )

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

    def save_best_model(self, X_train, y_train, dataset_size):

        best = select_best(self.results)

        best_name = best["Model"]

        save_json(best.to_dict(), MODEL_DIR / METRICS_FILENAME,)

        self.logger.info(  f"Best model selected: {best_name}" )

        trainer = self.trained_models[best_name]

        trainer.calibrate(
            X_train,
            y_train,
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

        baseline_pred = trainer.calibrated_pipeline.predict(
            X_test
        )

        baseline_prob = trainer.calibrated_pipeline.predict_proba(
            X_test
        )[:, 1]


        baseline_performance = ModelEvaluator.evaluate(
            trainer.calibrated_pipeline,
            X_test,
            y_test,
        )

        baseline_fairness = FairnessEvaluator.evaluate(
            y_true=y_test,
            y_pred=baseline_pred,
            y_prob=baseline_prob,
            sensitive_features=self.sensitive_test,
        )

        FairnessEvaluator.save(
            baseline_fairness,
            stage="baseline",
        )


        # ---------------------------------------------------------
        # Bias Mitigation
        # ---------------------------------------------------------

        self.logger.info("Applying Threshold Optimizer...")

        mitigator = BiasMitigator.threshold_optimizer(
            estimator=trainer.calibrated_pipeline,
            X_train=X_train,
            y_train=y_train,
            sensitive_features=self.sensitive_train,
            constraints=FAIRNESS_CONSTRAINT,
            objective=FAIRNESS_OBJECTIVE,
        )

        mitigated_pred = BiasMitigator.predict(
            mitigator,
            X_test,
            self.sensitive_test,
        )

        # ThresholdOptimizer does not change probabilities,
        # so we reuse the calibrated probabilities.
        mitigated_prob = baseline_prob

        mitigated_fairness = FairnessEvaluator.evaluate(
            y_true=y_test,
            y_pred=mitigated_pred,
            y_prob=mitigated_prob,
            sensitive_features=self.sensitive_test,
        )

        FairnessEvaluator.save(
            mitigated_fairness,
            stage="mitigated",
        )

        # ---------------------------------------------------------
        # Performance after mitigation
        # ---------------------------------------------------------

        mitigated_performance = {}

        mitigated_performance.update(

            ModelEvaluator.classification_metrics(
                y_test,
                mitigated_pred,
                mitigated_prob,
            )

        )

        mitigated_performance.update(

            ModelEvaluator.credit_metrics(
                y_test,
                mitigated_prob,
            )

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

        summary = FairnessComparison.summary(
            comparison
        )

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

        self.logger.info(
            "Responsible AI pipeline completed."
        )

        return comparison

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
            y_train,
            dataset_size=len(df),
        )

        self.logger.info(
            "Running Responsible AI pipeline"
        )

        self.responsible_ai_pipeline(
            trainer,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        self.logger.info(
            "Saving benchmark results"
        )

        self.save_results()

        self.logger.info(
            "Training completed successfully."
        )


if __name__ == "__main__":

    manager = TrainingManager()

    manager.run()