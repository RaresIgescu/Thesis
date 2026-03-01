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

        '''
        # The __init__ method initializes the environment by taking in the preprocessed features and labels,
        # as well as a maxSteps parameter that defines the maximum number of steps the agent can take in the environment before an episode ends.
        # The action space is defined as a discrete space with two possible actions (0 for normal traffic and 1 for attack),
        # and the observation space is defined as a continuous box with the same number of dimensions as the features, where each feature is scaled between 0 and 1 (because of the MinMaxScaler applied during preprocessing).
        '''
    
        super().__init__()
        self.features = features
        self.labels = labels
        self.maxSteps = maxSteps
        self.current_step = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(features.shape[1],), dtype=np.float32)

    def step(self, action):

        '''
        # The step functions defines the actual interaction between the agent and the environment. 
        # It takes an action as input, compares it to the true label of the current traffic instance
        # and assigns a reward based on whether the action was correct or not.
        # The reward structure is designed to encourage the agent to correctly identify normal traffic and attacks,
        # while also penalizing incorrect classifications more heavily for attacks than for normal traffic because
        # attacks are a lot less common and we want to make sure that the agent learns to identify them effectively.
        '''

        true_label = self.labels[self.current_step]
        reward = 0.0
        if(action == true_label):
            reward = 1.0
        elif(action == 0 and true_label == 1):
            reward = -1.0
        elif(action == 1 and true_label == 0):
            reward = -2.0

        # After the reward is calculated, we move to the next step in the environment by incrementing the current step counter.
        self.current_step += 1

        # We also check if we have reached the maximum number of steps allowed in the environment, which is defined by the maxSteps parameter.
        done = self.current_step >= self.maxSteps

        # If we did not reach the maximum number of steps, we return the next observation from the features dataset.
        if not done:
            next_observation = self.features[self.current_step]
        # Otherwise, we return a zero vector as the next observation, which indicates that the episode has ended and there are no more traffic instances to process.
        else:
            next_observation = np.zeros(self.features.shape[1])
            
        return np.array(next_observation, dtype=np.float32), reward, done, False, {}

    def reset(self, seed=None, options=None):

        '''
        # The reset function is responsible for moving the environment back to its initial state at the beginning of each episode.
        # It takes an optional seed parameter that can be used to ensure reproducibility of the environment
        # The reset function randomly selects a starting point in the features dataset for the new episode, 
        # ensuring that the agent is exposed to different traffic instances across episodes and can learn to generalize its behavior effectively.
        '''

        super().reset(seed=seed)
        max_limit = len(self.features) - self.maxSteps - 1
        self.current_step = np.random.randint(0, max_limit)
        first_observation = self.features[self.current_step]
        return np.array(first_observation, dtype=np.float32), {}

    def render(self):
        
        '''
        # The render function is a placeholder in this implementation, as we are not visualizing the environment in this project.
        '''

        pass

    def close(self):   

        '''
        # The close function is also a placeholder, as there are no resources to clean up in this implementation.
        '''

        pass

# === Define the agent ===

# === Evaluation === 