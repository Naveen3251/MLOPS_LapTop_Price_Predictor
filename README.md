# File Structure for the Zenml MLOPS of Project
You can maintain this structure for your any zenml related MLOps project

![mlops](https://github.com/Naveen3251/MLOPS_LapTop_Price_Predictor/assets/114800360/683401e2-779b-4947-b0fd-3e211d2667ef)

# Pre-requisites

### Python 3.7 or higher: 
Get it from here: [https://www.python.org/downloads/]

### Basic Zenml commands

#### Install zenml
```
pip install zenml
```

#### to Launch zenml server and dashboard locally
```
pip install "zenml[server]"
```

#### to see the zenml Version:
```
zenml version
```

#### To initiate a new repository
```
zenml init
```

#### to run the dashboard locally:
```
zenml up
```

#### to know the status of our zenml Pipelines
```
zenml show
```

### Integration of MLflow with ZenML

#### Integrating mlflow with ZenML
```
zenml integration install mlflow -y
```

#### Register the experiment tracker
###### syntax
```
zenml experiment-tracker register <mlflow_tracker_name> --flavor=mlflow
```
###### Example
```
zenml experiment-tracker register mlflow_tracker_employee --flavor=mlflow
```

#### Registering the model deployer
###### syntax
```
zenml model-deployer register <mlflow_deployername> --flavor=mlflow
```
###### Example
```
zenml model-deployer register mlflow_employee --flavor=mlflow
```

#### Registering the stack
###### syntax
```
zenml stack register <mlflowstackname> -a default -o default -d <registered_mlflowdeployername> -e  <registered_mlflowtrackername> --set
```
###### Example
```
zenml stack register mlflow_stack_employee -a default -o default -d mlflow_employee -e mlflow_tracker_employee --set
```
### Need to know
#### Introducing ZenML
ZenML is an open-source MLOPS Framework that helps to build portable and production-ready pipelines. The ZenML Framework will help us do this project using MLOPS.

#### Fundamental Concepts of MLOPS
  **Steps:** 
    Steps are single units of tasks in a pipeline or workflow. Each step represents a specific action or operation that needs to be performed to develop a machine-learning workflow. For example, data cleaning,       data preprocessing, training models, etc., are certain steps in developing a machine learning model.
  **Pipelines:**
    They connect multiple steps together to create a structured and automated process for machine learning tasks. for, e.g., the data processing pipeline, the model evaluation pipeline, and the model training        pipeline.

#### What is an Experiment Tracker?
An experiment tracker is a tool in machine learning used to record, monitor, and manage various experiments in the machine learning development process.

Data scientists experiment with different models to get the best results. So, they need to keep tracking data and using different models. It will be very hard for them if they record it manually using an Excel sheet.

#### MLflow
MLflow is a valuable tool for efficiently tracking and managing experiments in machine learning. It automates experiment tracking, monitoring model iterations, and associated data. This streamlines the model development process and provides a user-friendly interface for visualizing results.

Integrating MLflow with ZenML enhances experiment robustness and management within the machine learning operations framework.

#### Deployment
  ##### a). Continuous Deployment Pipeline

      This pipeline will automate the model deployment process. Once a model passes evaluation criteria, it’s automatically deployed to a production environment. For example, it starts with data preprocessing,         data cleaning, training the data, model evaluation, etc.<br>

  ##### b). Inference Deployment Pipeline

      The Inference Deployment Pipeline focuses on deploying machine learning models for real-time or batch inference. The Inference Deployment Pipeline specializes in deploying models for making predictions in        a production environment. For example, it sets up an API endpoint where users can send text. It ensures the model’s availability and scalability and monitors its real-time performance. These pipelines are        important for maintaining the efficiency and effectiveness of machine-learning systems.

#### How do we debug that the server daemon is not running?
This is a common error you will face in the project. Just run
```
zenml down
```
then
```
zenml disconnect
```
again run the pipeline. It will be resolved.

### Reference
### 1] MLOPS-Architecture design patterns
Some of common design patterns in Mlops architectures<br>
https://neptune.ai/blog/ml-pipeline-architecture-design-patterns
### 2] Mlflow
Refer my Mlflow github for basic understanding of mlflow<br>
https://github.com/Naveen3251/mlflow
