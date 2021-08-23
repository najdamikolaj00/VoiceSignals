'''
Program will show us means values in features from signal data split between sober and unsober.
'''
import numpy as np
import pandas as pd
import os

PARENT = os.path.dirname(os.getcwd())
FILE = os.path.join(PARENT, "VoiceSignals\\dataset01.csv")
data = pd.read_csv(FILE, header = 0)
data = data.drop(['filename'], axis=1)
print(data.groupby("label").mean())
# data.groupby("label").mean().to_csv('dataset01_mean.csv')
# data.groupby("label").std().to_csv('dataset01_std.csv')
