from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

from configs.config import (
    RANDOM_STATE,
    N_JOBS,
    N_JOBS2,
    LOGISTIC_MAX_ITER,
)

MODELS = {

    "logistic": LogisticRegression(
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_STATE,
    ),

    "random_forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS2,
    ),

    "extra_trees": ExtraTreesClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS2,
    ),

    "hist_gradient_boosting": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),

    "lightgbm": LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS2,
        verbosity=-1,
    ),

    "catboost": CatBoostClassifier(
        verbose=False,
        random_state=RANDOM_STATE,
    ),

    "ebm": ExplainableBoostingClassifier(
        random_state=RANDOM_STATE,
    ),
}