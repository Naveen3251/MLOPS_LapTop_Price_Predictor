from zenml import pipeline


from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.model_train import train_model
import logging

#Define a ZenML pipeline called training_pipeline.
@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    '''
    Data pipeline for training the model.
    '''
    #step ingesting data: returns the data.
    df = ingest_df(data_path=data_path)
    #step to clean the data.
    X_train, X_test, y_train, y_test = cleaning_data(df=df)
    #training the model
    model = train_model(X_train=X_train,y_train=y_train)