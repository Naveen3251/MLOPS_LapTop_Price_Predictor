from pipelines.training_pipeline import train_pipeline

from zenml.client import Client
if __name__ == '__main__':
    #printimg the experiment tracking uri
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    #Run the pipeline
    train_pipeline(data_path="C:\mlops_laptop\MLOPS_LapTop_Price_Predictor\data\laptop_data.csv")