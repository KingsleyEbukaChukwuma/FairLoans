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
)

from src.data.loader import DataLoader
from src.features.preprocessing import CreditPreprocessor

from src.models.registry import MODELS
from src.models.build_pipeline import build_pipeline
from src.models.trainer import ModelTrainer
from src.models.tune import ModelTuner
from src.models.evaluate import ModelEvaluator
from src.models.select_best import select_best
from src.models.save import (save_model, save_study, save_json, save_dataframe,)


class TrainingManager:

    def __init__(self):

        self.loader = DataLoader()
        self.preprocessor = CreditPreprocessor()

        self.results = []
        self.trained_models = {}

        self.sensitive_train = None
        self.sensitive_test = None

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

            print("=" * 60)
            print(f"Training {model_name}")
            print("=" * 60)

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

            print(metrics)

    def save_best_model(self, X_train, y_train):

        best = select_best(self.results)

        best_name = best["Model"]

        print(f"\nBest Model: {best_name}")

        trainer = self.trained_models[best_name]

        trainer.calibrate(
            X_train,
            y_train,
        )

        save_model(
            trainer.pipeline,
            MODEL_DIR / PIPELINE_FILENAME,
        )

    def save_results(self):

        save_dataframe(
            pd.DataFrame(self.results),
            REPORT_DIR / RESULTS_FILENAME,
        )

    def run(self):

        print("Loading data...")

        df = self.load_data()

        print("Splitting dataset...")

        (X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, ) = self.split_data(df)

        self.sensitive_train = sensitive_train
        self.sensitive_test = sensitive_test

        print("Training models...")

        self.train_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )



        print("Selecting best model...")

        self.save_best_model(
            X_train,
            y_train,
        )

        print("Saving results...")

        self.save_results()

        print("\nTraining Complete.")


if __name__ == "__main__":

    manager = TrainingManager()

    manager.run()