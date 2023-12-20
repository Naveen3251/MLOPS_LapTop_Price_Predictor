import logging

import pandas as pd
from src.clean_data import DataProcessing


def get_data_for_test():
    try:
        df=pd.read_csv("C:\mlops_laptop\MLOPS_LapTop_Price_Predictor\data\laptop_data.csv")

        df = df.sample(n=100)
        data_processing_strategy = DataProcessing()  # Replace with your actual class
        
        # Apply your custom data processing to the input DataFrame
        df_processed = data_processing_strategy.handle_data(df)

        result = df_processed.to_json(orient="split")
        return result
    except Exception as e:
        logging.error("Error during fetching test data : {}".format(e))
        raise e
