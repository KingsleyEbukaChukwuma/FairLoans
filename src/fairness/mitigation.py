from __future__ import annotations

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
)
from sklearn.base import clone


class BiasMitigator:
    """
    Apply Fairlearn bias mitigation techniques.
    """

    @staticmethod
    def threshold_optimizer(
        estimator,
        X_train,
        y_train,
        sensitive_features,
        objective: str = "accuracy_score",
        constraints: str = "demographic_parity",
    ):
        """
        Post-processing mitigation.

        Parameters
        ----------
        estimator
            Trained classifier.

        objective
            Usually:
                - accuracy_score
                - balanced_accuracy_score

        constraints
            demographic_parity
            equalized_odds
        """

        mitigator = ThresholdOptimizer(
            estimator=estimator,
            constraints=constraints,
            objective=objective,
            predict_method="predict_proba",
            prefit=True,
        )

        mitigator.fit(
            X_train,
            y_train,
            sensitive_features=sensitive_features,
        )

        return mitigator

    @staticmethod
    def exponentiated_gradient(
        estimator,
        X_train,
        y_train,
        sensitive_features,
        constraint: str = "demographic_parity",
    ):
        """
        In-processing mitigation.

        Parameters
        ----------
        constraint

            demographic_parity

            equalized_odds
        """

        if constraint == "demographic_parity":

            fairness_constraint = DemographicParity()

        elif constraint == "equalized_odds":

            fairness_constraint = EqualizedOdds()

        else:

            raise ValueError(f"Unsupported constraint: {constraint}")

        preprocessor = estimator.named_steps["preprocessor"]

        classifier = clone(estimator.named_steps["classifier"])

        X_train_processed = preprocessor.transform(
            X_train,
        )

        mitigator = ExponentiatedGradient(
            estimator=classifier,
            constraints=fairness_constraint,
        )

        mitigator.fit(
            X_train_processed,
            y_train,
            sensitive_features=sensitive_features,
        )

        return {
            "preprocessor": preprocessor,
            "mitigator": mitigator,
        }

    @staticmethod
    def predict(
        mitigated_model,
        X,
        sensitive_features=None,
    ):
        """
        Predict using a mitigated model.
        """

        # ThresholdOptimizer
        if isinstance(
            mitigated_model,
            ThresholdOptimizer,
        ):

            return mitigated_model.predict(
                X,
                sensitive_features=sensitive_features,
            )

        # ExponentiatedGradient
        X_processed = mitigated_model["preprocessor"].transform(
            X,
        )

        return mitigated_model["mitigator"].predict(
            X_processed,
        )
