Let's jump into the Python packages you need. Within the Python environment of your choice, run:

# Pre-requisites

### Python 3.7 or higher: 
Get it from here: [https://www.python.org/downloads/]

### Basic Zenml commands

#### Install zenml
```pip install zenml```

#### to Launch zenml server and dashboard locally
```pip install "zenml[server]"```

#### to see the zenml Version:
```zenml version```

#### To initiate a new repository
```zenml init```

#### to run the dashboard locally:
```zenml up```
or in windows if not supported use below command
```zenml up --blocking```

#### to know the status of our zenml Pipelines
```zenml show```

### Integration of MLflow with ZenML

#### Integrating mlflow with ZenML
```zenml integration install mlflow -y```

#### Register the experiment tracker
###### syntax
```zenml experiment-tracker register <mlflow_tracker_name> --flavor=mlflow```
###### Example
```zenml experiment-tracker register mlflow_tracker_employee --flavor=mlflow```

#### Registering the model deployer
###### syntax
```zenml model-deployer register <mlflow_deployername> --flavor=mlflow```
###### Example
```zenml model-deployer register mlflow_employee --flavor=mlflow```

#### Registering the stack
###### syntax
```zenml stack register <mlflowstackname> -a default -o default -d <registered_mlflowdeployername> -e  <registered_mlflowtrackername> --set```
###### Example
```zenml stack register mlflow_stack_employee -a default -o default -d mlflow_employee -e mlflow_tracker_employee --set```
