from xgboost import XGBClassifier

from mlapp.commons.configs import RANDOM_STATE
from mlapp.models.model_wrapper import ModelWrapper, ModelPhase
from mlapp.visualizations import helpers as vis

EVAL_METRIC = ["logloss", "aucpr"]


class XGBoostModel(ModelWrapper):
    DISPLAY_NAME = "XGBoost"
    SHORT_NAME = "xgb"

    _fit_result = None

    def __init__(self, training_approach, phase=ModelPhase.TRAINING, **kwargs):
        super().__init__(training_approach, phase=phase, **kwargs)

    @property
    def pristine_model(self):
        return XGBClassifier(verbosity=0)

    def fit_model(self):
        super().fit_model()
        try:
            # XGBClassifier default objective = "binary:logistic"
            model = XGBClassifier(**self._hyperparams, random_state=RANDOM_STATE)
            cv_scores = super().cross_validate(model)
            self._original_model = model.fit(
                self._dataset.X_train, self._dataset.y_train,
                eval_set=[(self._dataset.X_train, self._dataset.y_train), (self._dataset.X_test, self._dataset.y_test)],
                # how many iterations we will wait for the next decrease in the loss value
                early_stopping_rounds=self._hyperparams["patience"],
                eval_metric=EVAL_METRIC,
                verbose=0
            )
            self._fit_result = self._original_model.evals_result()
        except Exception as e:
            raise ValueError(e)

        return cv_scores

    def plt_learning_curves(self):
        vis.plt_xgb_learning_curve(
            self._fit_result,
            eval_metric=EVAL_METRIC,
            best_iteration=self._original_model.best_iteration,
            file_location=self._performance_dir
        )
