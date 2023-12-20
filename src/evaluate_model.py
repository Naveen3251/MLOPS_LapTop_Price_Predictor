import logging
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
from abc import ABC, abstractmethod
import numpy as np

# Abstract class for model evaluation
class Evaluate(ABC):
    @abstractmethod
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Abstract method to evaluate a machine learning model's performance.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Evaluation result.
        """
        pass

# Class to calculate Mean Absolute Error
class MeanAbsoluteError(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
            logging.info("Mean Absolute Error:", mae)
            return mae
        except Exception as e:
            logging.error("Error in calculating Mean Absolute Error", e)
            raise e

# Class to calculate R-squared (R2) score
class RSquaredScore(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info("R-squared (R2) score:", r2)
            return r2
        except Exception as e:
            logging.error("Error in calculating R-squared (R2) score", e)
            raise e

# Class to calculate Root Mean Square Error (RMSE)
class RootMeanSquareError(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
            logging.info("Root Mean Square Error (RMSE):", rmse)
            return rmse
        except Exception as e:
            logging.error("Error in calculating Root Mean Square Error (RMSE)", e)
            raise e
