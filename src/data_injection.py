import numpy as np
import pandas as pd

def load():
    df=pd.read_excel(r'https://raw.githubusercontent.com/Ashishswami99/AQI_prediction_model/main/data/AirQualityUCI.xlsx')

    return df