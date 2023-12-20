import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from abc import abstractmethod, ABC
import logging

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DataProcessing(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Drop Unnamed column
            df.drop('Unnamed: 0', axis=1, inplace=True)

            # Process RAM and Weight columns
            df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
            df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

            # Process ScreenResolution
            temp = df['ScreenResolution'].str.split('x', expand=True)
            df['Y_Res'] = temp[1].astype('int32')
            df['X_res'] = temp[0].str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0]).astype('int32')

            # Feature Engineering
            df['PPI'] = ((((df['X_res']**2) + (df['Y_Res']**2))**0.5) / df['Inches']).astype('float32')
            df['TouchScreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
            df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
            df['Cpu_Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

            # Preprocessing
            df['Cpu_brand'] = df['Cpu_Name'].apply(self._classify_cpu_brand)

            # Memory Processing
            df['Memory'] = df['Memory'].str.replace(r'\.0', '')
            df['Memory'] = df['Memory'].str.replace('GB', '')
            df['Memory'] = df['Memory'].str.replace('TB', '000')
            new = df['Memory'].str.split('+', n=1, expand=True)
            df['first'] = new[0].str.strip()
            df['second'] = new[1].fillna('0').str.strip()

            # Segregate SSD, HDD, HYBRID, flash storage for first column
            df = self._segregate_storage(df, 'first', 'Layer1')

            # Segregate SSD, HDD, HYBRID, flash storage for the second column
            df = self._segregate_storage(df, 'second', 'Layer2')

            logging.info('hdd',df['first'],df['second'])

            #to take only number from string
            df['first']=df['first'].str.extract('(\d+)')
            df['second']=df['second'].str.extract('(\d+)')

            logging.info('hdd',df)
            #converting type to int
            df['first']=df['first'].astype('int32')
            df['second']=df['second'].astype('int32')

            
            
            # Process HDD, SSD, Hybrid, FlashStorage
            df['HDD']=(df['first']*df['Layer1_HDD'])+(df['second']*df['Layer2_HDD'])
            df['SSD']=(df['first']*df['Layer1_SSD'])+(df['second']*df['Layer2_SSD'])
            df['Hybrid']=(df['first']*df['Layer1_Hybrid'])+(df['second']*df['Layer2_Hybrid'])
            df['FlashStorage']=(df['first']*df['Layer1_Flash_storage'])+(df['second']*df['Layer2_Flash_storage'])
            # Drop unnecessary columns
            df.drop(['first', 'second', 'Layer1_HDD', 'Layer1_SSD', 'Layer1_Hybrid', 'Layer1_Flash_storage',
                     'Layer2_HDD', 'Layer2_SSD', 'Layer2_Hybrid', 'Layer2_Flash_storage', 'Memory','Hybrid','FlashStorage'], axis=1, inplace=True)
            
            # Segregating Gpu based on brands
            df['Gpu Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
            df = df[df['Gpu Brand'] != 'ARM']

            # Operating System Processing
            df['os'] = df['OpSys'].apply(self._classify_os)
            df.drop(['OpSys'], axis=1, inplace=True)

            # Drop unnecessary columns
            df.drop(['ScreenResolution', 'Y_Res', 'X_res', 'Cpu_Name', 'Cpu', 'Inches'], axis=1, inplace=True)

            # Reorder columns
            df = df[['Company', 'TypeName', 'Ram', 'Weight', 'Price', 'TouchScreen',
                     'IPS', 'PPI', 'Cpu_brand', 'HDD', 'SSD', 'Gpu Brand', 'os']]

            return df

        except Exception as e:
            logging.error("Error in Data Processing", e)
            raise e

    def _classify_cpu_brand(self, text):
        if text in ['Intel Core i5', 'Intel Core i7', 'Intel Core i3']:
            return text
        elif text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD processor'

    def _segregate_storage(self, df, column, prefix):
        df[f'{prefix}_HDD'] = df[column].apply(lambda x: 1 if 'HDD' in x else 0)
        df[f'{prefix}_SSD'] = df[column].apply(lambda x: 1 if 'SSD' in x else 0)
        df[f'{prefix}_Hybrid'] = df[column].apply(lambda x: 1 if 'Hybrid' in x else 0)
        df[f'{prefix}_Flash_storage'] = df[column].apply(lambda x: 1 if 'Flash Storage' in x else 0)
        return df

    def _classify_os(self, text):
        if text in ['Windows 10', 'Windows 10 S', 'Windows 7']:
            return 'Windows'
        elif text in ['macOS', 'Mac OS X']:
            return 'Mac'
        else:
            return 'others/Linux/No OS'


class DataSplitting(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> tuple:
        try:
            X = df.drop('Price', axis=1)
            Y = np.log(df['Price'])
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Error in Data Splitting", e)
            raise e

