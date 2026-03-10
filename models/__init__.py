from mlapp.models.linear_svc import LinearSVCModel
from mlapp.models.logistic_regression import LogisticRegressionModel
from mlapp.models.mlp import MLPModel
from mlapp.models.model_wrapper import ModelWrapper, ModelPhase, skl_cross_validate
from mlapp.models.random_forest import RandomForestModel
from mlapp.models.svc import SVCModel
from mlapp.models.xgb import XGBoostModel

MODEL_OPTIONS = {
    LogisticRegressionModel.SHORT_NAME: LogisticRegressionModel,
    MLPModel.SHORT_NAME: MLPModel,
    RandomForestModel.SHORT_NAME: RandomForestModel,
    SVCModel.SHORT_NAME: SVCModel,
    LinearSVCModel.SHORT_NAME: LinearSVCModel,
    XGBoostModel.SHORT_NAME: XGBoostModel
}
