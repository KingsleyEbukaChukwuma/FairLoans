from __future__ import annotations

import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from src.configs.config import (
    CV_FOLDS,
    N_JOBS,
    N_JOBS2,
    OPTIMIZATION_METRIC,
    OPTUNA_DIRECTION,
    OPTUNA_RANDOM_STATE,
    OPTUNA_TIMEOUT,
)


class ModelTuner:

    def __init__(self, pipeline, X, y, cv=CV_FOLDS):

        self.pipeline = pipeline
        self.X = X
        self.y = y
        self.cv = cv

    def optimize(self, model_name, n_trials):

        study = optuna.create_study(
            direction=OPTUNA_DIRECTION,
            sampler=optuna.samplers.TPESampler(
                seed=OPTUNA_RANDOM_STATE,
            ),
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(
            lambda trial: self.objective(
                trial,
                model_name,
            ),
            n_trials=n_trials,
            timeout=OPTUNA_TIMEOUT,
            n_jobs=N_JOBS,
            gc_after_trial=True,
        )

        return study

    def objective(self, trial, model_name):

        pipeline = clone(self.pipeline)

        model = pipeline.named_steps["classifier"]

        if model_name == "logistic":

            params = {
                "C": trial.suggest_float(
                    "C",
                    1e-3,
                    100,
                    log=True,
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight",
                    [
                        None,
                        "balanced",
                    ],
                ),
            }

        elif model_name == "random_forest" or model_name == "extra_trees":

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    500,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    30,
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    2,
                    20,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    10,
                ),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    [
                        "sqrt",
                        "log2",
                        None,
                    ],
                ),
            }

        elif model_name == "hist_gradient_boosting":

            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3,
                ),
                "max_iter": trial.suggest_int(
                    "max_iter",
                    100,
                    500,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    20,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    10,
                    100,
                ),
            }

        elif model_name == "lightgbm":

            params = {
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    20,
                    150,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3,
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    500,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    12,
                ),
            }

        elif model_name == "catboost":

            params = {
                "depth": trial.suggest_int(
                    "depth",
                    3,
                    10,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3,
                ),
                "iterations": trial.suggest_int(
                    "iterations",
                    100,
                    500,
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg",
                    1,
                    10,
                ),
            }

        elif model_name == "ebm":

            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.005,
                    0.05,
                ),
                "max_rounds": trial.suggest_int(
                    "max_rounds",
                    200,
                    1000,
                ),
            }

        else:

            params = {}

        model.set_params(**params)

        scores = cross_val_score(
            pipeline,
            self.X,
            self.y,
            scoring=OPTIMIZATION_METRIC,
            cv=self.cv,
            n_jobs=N_JOBS2,
        )

        return scores.mean()
