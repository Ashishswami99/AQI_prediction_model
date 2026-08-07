import numpy as np
import pandas as pd

def load():
    df=pd.read_excel(r'C:/AQI_prediction_model/data/AirQualityUCI.xlsx')

    return df