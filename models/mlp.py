import gc
import os
import platform
from pathlib import PurePosixPath

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from keras import backend
from keras import optimizers, regularizers, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Activation, Dense, Dropout
from keras.metrics import Precision, Recall, AUC
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold

from mlapp import logger
from mlapp.bulk.configs import EARLY_STOPPING
from mlapp.commons.configs import RANDOM_STATE
from mlapp.models.model_wrapper import ModelWrapper, ModelPhase
from mlapp.visualizations import helpers as vis


class MLPModel(ModelWrapper):
    DISPLAY_NAME = "Multilayer Perceptron"
    SHORT_NAME = "mlp"

    _should_scale = True
    _predict_probability = True  # keras predicts probability
    _fit_result = None

    def __init__(self, training_approach, phase=ModelPhase.TRAINING, **kwargs):
        super().__init__(training_approach, phase=phase, **kwargs)
        # model data saves in a folder instead of a file
        # self._model_data_location = PurePosixPath(self._dataset.training_approach.model_dir) / self.SHORT_NAME
        self._model_data_location = PurePosixPath(
            self._dataset.training_approach.model_dir) / f"{self.SHORT_NAME}.keras"

    def cross_validate(self, _):
        X, y = self._dataset.get_xy_dataset()
        metadata = self._dataset.training_approach.training_metadata
        cv_folds = metadata["cv_folds"]
        cv_metric = metadata["cv_metric"]
        batch_size = self._hyperparams["batch_size"]
        epochs = self._hyperparams["epochs"]

        # get cv score
        _, fold_score_list = MLPModel.mlp_cross_validate(
            self, X, y,
            cv_folds=cv_folds, cv_metric=cv_metric, batch_size=batch_size, epochs=epochs, kwargs=self._hyperparams
        )
        logger.log_info(msg=f"mlp {cv_metric} fold_score_list: {fold_score_list}")

        MLPModel.clean_up_mem()

        self._cv_scores = np.array(fold_score_list)
        return self._cv_scores

    def fit_model(self):
        super().fit_model()

        batch_size = self._hyperparams["batch_size"]
        epochs = self._hyperparams["epochs"]
        patience = self._hyperparams["patience"]
        monitor = MLPModel.generate_monitor(patience)

        model = self.build_model(**self._hyperparams)

        self._fit_result = model.fit(
            self._dataset.X_train, self._dataset.y_train,
            validation_data=(self._dataset.X_test, self._dataset.y_test),
            verbose=0,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[monitor]
        )
        logger.log_trace(msg=f"fit_result.history.keys(): {self._fit_result.history.keys()}")

        self._original_model = model
        cv_scores = self.cross_validate(None)

        # clean up to prevent memory leaks
        del model
        MLPModel.clean_up_mem()

        return cv_scores

    def save(self):
        self._original_model.save(self._model_data_location)

    def load(self):
        self._original_model = load_model(self._model_data_location)
        return self

    def get_proba_pred(self):
        return self._reshape_array(self._y_pred)

    def plt_learning_curves(self):
        vis.plt_mlp_learning_curve(self._fit_result.history, self._performance_dir)

    def build_model(self, **kwargs):
        input_dim = self._dataset.X_train.shape[1]
        dropout = kwargs["dropout"]
        optimizer = MLPModel.define_optimizer(kwargs["optimizer"], kwargs["learning_rate"])
        regularizer = MLPModel.define_regularizer(kwargs["l1"], kwargs["l2"])

        if "hidden_layers" in kwargs:
            hidden_layers = kwargs["hidden_layers"]
        else:
            percentage = kwargs["percentage"]
            shrink = kwargs["shrink"]
            hidden_layers = MLPModel.build_dynamic_layers(input_dim, percentage, shrink)

        mlp = Sequential()
        for idx, num in enumerate(hidden_layers):
            if idx == 0:  # first hidden layer
                mlp.add(Dense(
                    hidden_layers[0],
                    activation="relu",
                    input_dim=input_dim,
                    activity_regularizer=regularizer
                ))
            else:  # the rest hidden layer
                mlp.add(Dense(num, activation=None))  # no activation yet
                # do BatchNormalization before applying activation
                mlp.add(BatchNormalization())  # normalize activations
                mlp.add(Activation("relu"))  # then apply activation

            mlp.add(Dropout(dropout))

        mlp.add(Dense(1, activation="sigmoid"))  # output layer
        mlp.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc")
            ]  # no support for f1
        )
        logger.log_trace(msg=f"mlp summary: {mlp.summary()}")

        return mlp

    @staticmethod
    def _reshape_array(predictions):
        proba_pred = np.zeros(0)
        x_len = 0
        for probability in predictions:
            probability = np.insert(probability, 0, 1 - probability[0])
            proba_pred = np.append(proba_pred, probability)
            x_len = int(proba_pred.size / 2)  # decide shape x
        return proba_pred.reshape(x_len, 2)

    @staticmethod
    def generate_monitor(patience):
        return EarlyStopping(
            # Available metrics are: loss,accuracy,precision,recall,auc. Always use loss to train mlp
            monitor="loss",
            min_delta=1e-3,
            patience=patience,
            verbose=1,
            mode="auto",
            restore_best_weights=True
        )

    @staticmethod
    def define_regularizer(l1: float, l2: float):
        if l1 > 0 and l2 > 0:
            regularizer = regularizers.l1_l2(l1=l1, l2=l2)
        elif l1 > 0:
            regularizer = regularizers.l1(l1)
        elif l2 > 0:
            regularizer = regularizers.l2(l2)
        else:
            regularizer = None
        return regularizer

    @staticmethod
    def define_optimizer(optimizer_str, lr):
        if platform.processor() == "arm":
            if optimizer_str == "SGD":
                optimizer = optimizers.legacy.SGD(learning_rate=lr)
            elif optimizer_str == "Adam":
                optimizer = optimizers.legacy.Adam(learning_rate=lr)
            else:
                raise ValueError(f"optimizer {optimizer_str} not defined yet")
        else:
            if optimizer_str == "SGD":
                optimizer = optimizers.SGD(learning_rate=lr)
            elif optimizer_str == "Adam":
                optimizer = optimizers.Adam(learning_rate=lr)
            elif optimizer_str == "Adamax":
                optimizer = optimizers.Adamax(learning_rate=lr)
            else:
                raise ValueError(f"optimizer {optimizer_str} not defined yet")
        return optimizer

    @staticmethod
    def build_dynamic_layers(input_dim, percentage, shrink):
        neuron_num = int(percentage * input_dim)
        layers = []

        while neuron_num > 25 and len(layers) < 3:
            layers.append(neuron_num)
            neuron_num = int(neuron_num * shrink)
        return layers

    @staticmethod
    def mlp_cross_validate(model_wrapper, X, y, cv_folds, cv_metric, batch_size, epochs, kwargs):
        # StratifiedKFold split
        skf = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
        epoch_list = []  # which epoch it stopped for this round
        fold_score_list = []
        splits = list(skf.split(X, y))

        if os.getenv("CUDA_VISIBLE_DEVICES") != "-1" and tf.config.list_physical_devices("GPU"):  # use gpu
            for train_index, test_index in skf.split(X, y):
                epoch, score = train_by_fold(
                    X, y, train_index, test_index, model_wrapper, cv_metric, batch_size, epochs, kwargs)
                epoch_list.append(epoch)
                fold_score_list.append(score)

            return epoch_list, fold_score_list
        else:  # use cpu with parallel
            # n_jobs means how many folds are trained in parallel
            results = Parallel(n_jobs=cv_folds)(delayed(train_by_fold)(
                X, y, train_index, test_index, model_wrapper, cv_metric, batch_size, epochs, kwargs
            ) for train_index, test_index in splits)

            # unpack results
            epoch_tuples, fold_score_tuples = zip(*results)
            return list(epoch_tuples), list(fold_score_tuples)

    # clean up to prevent memory leaks:
    # Each trial in Ray Tune can run in a separate process. Without proper cleanup:
    # 	• TensorFlow may hold onto GPU memory across trials.
    # 	• Models and graphs can accumulate in RAM.
    # 	• Trials may crash due to OutOfMemory (OOM) errors.
    @staticmethod
    def clean_up_mem():
        backend.clear_session()
        gc.collect()


def train_by_fold(X, y, train_index, test_index, model_wrapper, cv_metric, batch_size, epochs, kwargs):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = model_wrapper.build_model(**kwargs)
    monitor = MLPModel.generate_monitor(EARLY_STOPPING)

    model.fit(
        X_train, y_train,
        # validation_data=(X_test, y_test), # change to use evaluate()
        verbose=0,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[monitor]
    )
    loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test, verbose=0)

    # depending on cv_metric, we get the matched value from the dictionary, default to None if not found
    score = {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }.get(cv_metric, None)

    # clean up to prevent memory leaks
    del model, X_train, X_test, y_train, y_test

    return monitor.stopped_epoch, score
