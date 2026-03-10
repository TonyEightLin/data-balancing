import ast
import os
import pickle
from enum import Enum
from pathlib import PurePosixPath

from joblib import parallel_backend
from mlapp import logger
from mlapp.commons import utils
from mlapp.commons.configs import N_JOBS, RANDOM_STATE
from mlapp.commons.exceptions import PredictionNotAppliedError
from mlapp.datatasks import MLDataset
from mlapp.visualizations import helpers as vis
from munch import Munch
from sklearn import metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold


class ModelPhase(Enum):
    TRAINING = "training"
    PARAM_SEARCHING = "param-searching"
    PREDICTION = "prediction"
    ACD = "acd"


class ModelWrapper:
    DISPLAY_NAME = None
    SHORT_NAME = None

    _should_scale = False
    _original_model = None  # the original ml model, e.g. from sklearn, keras, etc...
    _predict_probability = False
    _y_pred = None
    _y_pred_binary = None
    _y_score = None
    _cv_scores = None
    _prediction_data = None
    _hyperparams = None

    def __init__(self, training_approach, phase, **kwargs):
        self._phase = phase
        self._dataset = MLDataset(training_approach, phase=self._phase, should_scale=self._should_scale)

        self._path_check()

        if self._phase in [ModelPhase.TRAINING, ModelPhase.PARAM_SEARCHING]:
            self._init_training(**kwargs)

    def _path_check(self):
        """Make sure directories and filepaths exist."""
        ta = self._dataset.training_approach

        model_dir = ta.model_dir
        if model_dir:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        else:
            raise ValueError("please specify model_dir")
        self._model_data_location = PurePosixPath(model_dir) / f"{self.SHORT_NAME}.pkl"

        if self._phase == ModelPhase.TRAINING:
            self._performance_dir = ta.performance_dir
            if self._performance_dir:
                if not os.path.exists(self._performance_dir):
                    os.makedirs(self._performance_dir)
            else:
                raise ValueError("please specify performance_dir")

            if not ta.test_result_filepath:
                raise ValueError("please specify test_result_filepath")
            if not ta.training_data_filepath:
                raise ValueError("please specify training_data_filepath")

        if self._phase == ModelPhase.PREDICTION:
            if not ta.prediction_result_filepath:
                raise ValueError("please specify prediction_result_filepath")

    def _init_training(self, **kwargs):
        """Initialize training necessary data."""
        # generate training_data if it doesn't exist
        if not os.path.exists(self._dataset.training_approach.training_data_filepath):
            self._dataset.generate_training_data()

        self._dataset.split_dataset()
        self._dataset.handle_imbalance()
        self._dataset.handle_imbalance(under_sampling=False) # under_sampling

        # scale data if necessary
        if self._should_scale:
            self._dataset.scale_train_data()

        # hyperparams can be assigned in model declaration, or read from training_metadata
        if self.phase == ModelPhase.PARAM_SEARCHING:  # no need to set hyperparams
            pass
        elif "hyperparams" in kwargs:
            self._hyperparams = kwargs["hyperparams"]
        else:
            self._hyperparams = Munch.toDict(
                self._dataset.training_approach.training_metadata["hyperparams"][self.SHORT_NAME])

        # for value is 'None' from JSON, convert it to real Python None here
        if self._hyperparams:
            for key, value in self._hyperparams.items():
                if value == 'None':
                    self._hyperparams[key] = ast.literal_eval(value)

        if self.phase != ModelPhase.PARAM_SEARCHING and not self._hyperparams:
            raise ValueError(f"please assign params/hyper-params for model {self.DISPLAY_NAME}")

        if self._hyperparams:
            logger.log_info(msg=f"hyper-params for {self.SHORT_NAME}: {self._hyperparams}")

    @property
    def dataset(self):
        """Return ML dataset initialized in the model."""
        return self._dataset

    @property
    def pristine_model(self):
        """Return original ml model without any params set."""
        return None

    @property
    def phase(self):
        """Return if this model phase is for training, evaluation, or prediction."""
        return self._phase

    @property
    def original_model(self):
        """Return the original ml model wrapped in this class.
        It'll be fitted model if it's for training, or loaded model if it's for prediction or evaluation."""
        return self._original_model

    @property
    def model_data_location(self):
        """Return where the model is saved."""
        return self._model_data_location

    @property
    def y_pred_binary(self):
        """Return predictions in binary form."""
        return self._y_pred_binary

    @property
    def y_score(self):
        """Return predictions in probability form."""
        return self._y_score

    @property
    def cv_scores(self):
        """Return the cross validation score for all the folds."""
        return self._cv_scores

    def cross_validate(self, model):
        X, y = self._dataset.get_xy_dataset()
        metadata = self._dataset.training_approach.training_metadata
        cv_folds = metadata["cv_folds"]
        cv_metric = metadata["cv_metric"]
        self._cv_scores = skl_cross_validate(model, X, y, cv_folds, cv_metric)

        return self._cv_scores

    def fit_model(self):  # abstract function that should be implemented
        if self.phase != ModelPhase.TRAINING:
            raise PredictionNotAppliedError("fit_model")
        logger.log_info(msg=f"fitting {self.SHORT_NAME}...")

    def evaluate(self):
        if self.phase != ModelPhase.TRAINING:
            raise PredictionNotAppliedError("evaluate")

        # if _y_pred is None, execute predict() to obtain
        if self._y_pred is None:
            self.predict()

        # define y_pred_binary and y_pred_probability
        if self._predict_probability:
            # for keras case, y_pred changes from probability to binary for evaluation
            self._y_pred_binary = self._y_pred > 0.5
            self._y_score = self._y_pred
        else:
            self._y_pred_binary = self._y_pred
            try:
                # pay attention that sklearn predict_proba() can be different from predict()
                probability_pair = self._original_model.predict_proba(self._dataset.X_test)
                self._y_score = probability_pair[:, 1]
            except AttributeError:  # 'LinearSVC' object has no attribute 'predict_proba'
                self._y_score = self._y_pred

        y_true = self._dataset.y_test

        performances = {
            "confusion_matrix": f"\n{metrics.confusion_matrix(y_true, self._y_pred_binary)}",
            "classification_report": f"\n{metrics.classification_report(y_true, self._y_pred_binary)}",
            "accuracy_score": utils.to_decimal(metrics.accuracy_score(y_true, self._y_pred_binary)),
            # Use zero_division parameter: Precision is ill-defined and being set to 0.0 due to no predicted samples.
            "precision_score": utils.to_decimal(metrics.precision_score(y_true, self._y_pred_binary, zero_division=0)),
            "mcc_score": utils.to_decimal(metrics.matthews_corrcoef(y_true, self._y_pred_binary)),
            "roc_auc_score": utils.to_decimal(metrics.roc_auc_score(y_true, self._y_score))
        }

        [logger.log_info(msg=f"{key} -> {value}") for key, value in performances.items()]
        return performances

    def save(self):
        """Pickle the model."""
        if self.phase != ModelPhase.TRAINING:
            raise PredictionNotAppliedError("save")

        logger.log_info(msg=f"saving {self.SHORT_NAME}")
        with open(self._model_data_location, "wb") as files:
            pickle.dump(self._original_model, files)

    def load(self):
        """Load the model and return model wrapper."""
        with open(self._model_data_location, "rb") as f:
            self._original_model = pickle.load(f)
        return self

    def predict(self):
        if self.phase in [ModelPhase.PREDICTION, ModelPhase.ACD]:  # will get loaded model
            self.load()
            X = self._dataset.prediction_data
        else:  # will get fitted model, ml_model has been set when fitting
            X = self._dataset.X_test

        self._y_pred = self._original_model.predict(X)
        return self._y_pred  # _y_pred can be binary or probability

    def plt_confusion_matrix(self):
        if self.phase != ModelPhase.TRAINING:
            raise PredictionNotAppliedError("plt_confusion_matrix")

        vis.plt_confusion_matrix(
            self.DISPLAY_NAME,
            self.SHORT_NAME,
            y_test=self._dataset.y_test,
            y_pred_binary=self._y_pred_binary,
            file_location=self._performance_dir
        )

    def plt_learning_curves(self):
        pass


def skl_cross_validate(model, X, y, cv_folds, cv_metric):
    with parallel_backend("threading", n_jobs=N_JOBS):
        result = cross_val_score(
            model,
            X, y,  # cross validate the whole dataset
            cv=StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring=cv_metric,
            # https://scikit-learn.org/stable/modules/grid_search.html#robustness-to-failure
            # with scikit-learn = "==1.0.2"
            error_score=0
        )
    return result
