import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
from zenml import step
from sklearn.model_selection import train_test_split
from src.clean_data import DataProcessing, DataSplitting  # Replace with your actual module
import logging

# Define a ZenML step for data processing and splitting
@step(enable_cache=False)
def cleaning_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        # Instantiate your custom data processing strategy
        data_processing_strategy = DataProcessing()  # Replace with your actual class
        
        # Apply your custom data processing to the input DataFrame
        df_processed = data_processing_strategy.handle_data(df)
        
        # Instantiate your custom data splitting strategy
        data_splitting_strategy = DataSplitting()  # Replace with your actual class
        
        # Split the processed data into training and testing sets
        X_train, X_test, y_train, y_test = data_splitting_strategy.handle_data(df_processed)
        
        # Return the split data as a tuple
        return X_train, X_test, y_train, y_test
    except Exception as e:
        # Handle and log any errors that occur during data processing and splitting
        logging.error("Error in data processing and splitting step", e)
        raise e
