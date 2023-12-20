import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from abc import ABC, abstractmethod
import logging

# Abstract model
class Model(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model on given data
        """
        pass

# Gradient Boosting Regressor model
class GradientBoostingReg(Model):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Training the Gradient Boosting Regressor model
        
        Args:
            X_train: pd.DataFrame,
            y_train: pd.Series
        """
        step1 = ColumnTransformer(transformers=[
            ('col_inf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11])
        ], remainder='passthrough')
        logging.info("Model Training started")
        logging.info("data",X_train)
        gr_reg = GradientBoostingRegressor(n_estimators=500)
        pipe_gr = Pipeline([
            ('step1', step1),
            ('step2', gr_reg)
        ])
        pipe_gr.fit(X_train, y_train)
        logging.info("Model training finished")
        return pipe_gr
