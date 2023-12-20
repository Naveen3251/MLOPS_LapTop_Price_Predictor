import numpy as np
import json
import logging
import pandas as pd


from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output


from .utils import get_data_for_test

from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.data_cleaning import cleaning_data
from steps.evaluation import evaluate_model



# Define Docker settings with MLflow integration
docker_settings = DockerSettings(required_integrations = {MLFLOW})


#Define class for deployment pipeline configuration

"""Continuous"""
class DeploymentTriggerConfig(BaseParameters):
    min_accuracy:float = 0.75

@step 
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """
    It trigger the deployment only if accuracy is greater than min accuracy.
    Args:
        accuracy: accuracy of the model.
        config: Minimum accuracy thereshold.
    """
    try:
        return accuracy >= config.min_accuracy
    except Exception as e:
        logging.error("Error in deployment trigger",e)
        raise e

# Define a continuous pipeline
@pipeline(enable_cache=False,settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    data_path:str,
    min_accuracy:float = 0.75,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
  
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = cleaning_data(df=df)
    
    model = train_model(X_train=X_train, y_train=y_train)
   
    mae_score_result, r2_score_result, rmse_score_result = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    
    deployment_decision = deployment_trigger(accuracy=r2_score_result)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout
    )

"""Inference"""  
class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step
def predictor(service: MLFlowDeploymentService, data: str) -> np.ndarray:
    service.start(timeout=10)
    data = json.loads(data)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    logging.info("***********inference****************")
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=model_deployment_service, data=batch_data)
    return prediction