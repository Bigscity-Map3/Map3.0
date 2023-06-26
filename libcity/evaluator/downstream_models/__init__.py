from libcity.evaluator.downstream_models.regression_model import RegressionModel
from libcity.evaluator.downstream_models.kmeans_model import KmeansModel
from libcity.evaluator.downstream_models.abstract_model import AbstractModel
from libcity.evaluator.downstream_models.speed_inference import SpeedInferenceModel
from libcity.evaluator.downstream_models.time_estimation import TimeEstimationModel
from libcity.evaluator.downstream_models.loc_classification import LocClassificationModel
from libcity.evaluator.downstream_models.nextloc_pre import NextLocPreModel
__all__ = [
    "RegressionModel",
    "KmeansModel",
    "AbstractModel",
    "SpeedInferenceModel",
    "TimeEstimationModel",
    "LocClassificationModel",
    "NextLocPreModel"
]