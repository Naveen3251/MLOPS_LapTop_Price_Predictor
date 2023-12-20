import logging
 
import pandas as pd
from src.training_model import GradientBoostingReg
from zenml import step
from .config import ModelName

from sklearn.pipeline import Pipeline
#import 
from zenml.client import Client
import mlflow

# Obtain the active stack's experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker


#Define a step called train_model
@step(experiment_tracker = experiment_tracker.name,enable_cache=False)
def train_model(X_train:pd.DataFrame,y_train:pd.Series,config:ModelName)->Pipeline:
    """
    Trains the data based on the configured model
        
    """
    try:
        model = None
        if config.model_name == "GradientBoostingRegressor":
            mlflow.sklearn.autolog()
            model = GradientBoostingReg()
        else:
            raise ValueError("Model name is not supported")
        logging.info("data",X_train)
        trained_model = model.train(X_train=X_train,y_train=y_train)
        logging.info("Training model completed.")
        return trained_model
    
    except Exception as e:
        logging.error("Error in step training model",e)
        raise e