import logging
import pandas as pd
import numpy as np
from zenml import step
from src.evaluate_model import (
    MeanAbsoluteError,RSquaredScore,RootMeanSquareError
)
from typing import Tuple
from typing_extensions import Annotated
from sklearn.pipeline import Pipeline

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache = False)
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "meanabsolute_score"],
    Annotated[float, "r2_score"],
    Annotated[float, "rootmeansquare_score"]
]:
    """
    Evaluate a machine learning model's performance using common metrics.
    """
    try:
        y_pred = model.predict(X_test)

        # MAE Score
        mae_score_class = MeanAbsoluteError()
        mae_score_result = mae_score_class.evaluate_model(
            y_true=y_test, y_pred=y_pred
        )
        logging.info("MeanAbsolute Score:", mae_score_result)

        # R2 Score
        r2_score_class = RSquaredScore()
        r2_score_result = r2_score_class.evaluate_model(
            y_true=y_test, y_pred=y_pred
        )
        logging.info("R2 Score:", r2_score_result)

        # RMSE Score
        rmse_score_class = RootMeanSquareError()
        rmse_score_result = rmse_score_class.evaluate_model(
            y_true=y_test, y_pred=y_pred
        )
        logging.info("RMSE Score:", rmse_score_result)

        return  mae_score_result, r2_score_result, rmse_score_result

    except Exception as e:
        logging.error("Error in evaluating model", e)
        raise e
