from sklearn.linear_model import LogisticRegression

from mlapp.commons.configs import RANDOM_STATE
from mlapp.models.model_wrapper import ModelWrapper, ModelPhase


class LogisticRegressionModel(ModelWrapper):
    DISPLAY_NAME = "Logistic Regression"
    SHORT_NAME = "lr"

    def __init__(self, training_approach, phase=ModelPhase.TRAINING, **kwargs):
        super().__init__(training_approach, phase=phase, **kwargs)

    @property
    def pristine_model(self):
        return LogisticRegression()

    def fit_model(self):
        super().fit_model()
        try:
            model = LogisticRegression(**self._hyperparams, random_state=RANDOM_STATE)
            cv_scores = super().cross_validate(model)
            self._original_model = model.fit(self._dataset.X_train, self._dataset.y_train)
        except Exception as e:
            raise ValueError(e)

        return cv_scores
