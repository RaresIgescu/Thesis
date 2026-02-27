import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import time

# === LOAD AND PREPROCESS DATA ===
# In order for the AI agent to actually learn different attack types,
# we need to preprocess the traffic data and convert it into a numerical format.

# Load the CICIDS2017 dataset 
data = pd.read_csv('code/data/cicids2017_cleaned.csv')

# Remove any hidden spaces from column names
data.columns = data.columns.str.strip()

# Replace the infinite values with NaN and then drop any rows that contain NaN values to ensure that our dataset is clean and ready for training
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Encode the labels in a binary format: 0 for normal traffic and 1 for any kind of attack 
answer = data.loc[:, 'Attack Type'].apply(lambda x: 0 if x == 'Normal Traffic' else 1) 

# Drop the last column named 'Attack Type'
inputData = data.drop('Attack Type', axis=1)

# We need to scale the input data to ensure that all features are on the same scale, which can help the AI agent learn more effectively.
scaler = MinMaxScaler()
inputData_scaled = scaler.fit_transform(inputData)

# Print the distribution of attack types and the range of the scaled input data to verify that the preprocessing steps have been applied correctly.
# The first print statement will show us how many instances of normal traffic and different attack types are present in the dataset, 
# While the second print statement will confirm that the input data has been scaled to a range between 0 and 1.
print(data.loc[:, 'Attack Type'].value_counts())
print(inputData_scaled.min(), inputData_scaled.max())

# === DEFINE THE ENVIRONMENT ===
class IntrustionEnv(gym.Env):
    
    metadata = {"render_modes": []}

    def __init__(self, features, labels, maxSteps=1000):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...
# === Define the agent ===

# === Evaluation === 