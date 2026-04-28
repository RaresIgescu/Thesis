import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

import joblib

SEED = 42
np.random.seed(SEED)

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
answer = data.loc[:, 'Attack Type'].apply(lambda x: 0 if x == 'Normal Traffic' else 1).values

# Drop the last column named 'Attack Type'
inputData = data.drop('Attack Type', axis=1)

# Split into train/test before scaling to avoid data leakage.
# stratify=answer ensures both splits have the same class distribution.
train_features, test_features, train_labels, test_labels = train_test_split(
    inputData.values, answer, test_size=0.2, random_state=42, stratify=answer
)

# We need to scale the input data to ensure that all features are on the same scale, which can help the AI agent learn more effectively.
# The scaler is fitted only on the training set; the test set is transformed without fitting to prevent leakage.
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled  = scaler.transform(test_features)
scaler.feature_names_in_ = None  # suppress feature-name warnings during real-time inference

# Print the distribution of attack types and the range of the scaled input data to verify that the preprocessing steps have been applied correctly.
# The first print statement will show us how many instances of normal traffic and different attack types are present in the dataset,
# While the second print statement will confirm that the input data has been scaled to a range between 0 and 1.
print(data.loc[:, 'Attack Type'].value_counts())
print(train_features_scaled.min(), train_features_scaled.max())

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
        self.steps_taken = 0
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
            reward = -2.0
        elif(action == 1 and true_label == 0):
            reward = -1.3

        # After the reward is calculated, we move to the next step in the environment by incrementing the current step counter.
        self.current_step += 1
        self.steps_taken += 1

        # We also check if we have reached the maximum number of steps allowed in the environment, which is defined by the maxSteps parameter.
        done = self.steps_taken >= self.maxSteps

        # If we did not reach the maximum number of steps, we return the next observation from the features dataset.
        if not done:
            next_observation = self.features[self.current_step]
        # Otherwise, we return a zero vector as the next observation, which indicates that the episode has ended and there are no more traffic instances to process.
        else:
            next_observation = np.zeros(self.features.shape[1])
            
        return np.array(next_observation, dtype=np.float32), reward, False, done, {}

    def reset(self, seed=None, options=None):

        '''
        # The reset function is responsible for moving the environment back to its initial state at the beginning of each episode.
        # It takes an optional seed parameter that can be used to ensure reproducibility of the environment
        # The reset function randomly selects a starting point in the features dataset for the new episode, 
        # ensuring that the agent is exposed to different traffic instances across episodes and can learn to generalize its behavior effectively.
        '''

        super().reset(seed=seed)
        rng = self.np_random  # seeded RNG provided by Gymnasium
        # Force ~50% of episodes to start on an attack sample
        if rng.random() < 0.5:
            attack_indices = np.where(self.labels == 1)[0]
            self.current_step = rng.choice(attack_indices[:-self.maxSteps])
        else:
            max_limit = len(self.features) - self.maxSteps - 1
            self.current_step = rng.integers(0, max_limit)
        self.steps_taken = 0
        return np.array(self.features[self.current_step], dtype=np.float32), {}

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

# === DEFINE THE AGENT ===

# Considering that we have defined the environment in which the agent will be trained, we can now define the agent itself.
# We will be using the DQN algorithm from the stable_baselines3 library, which is a popular reinforcement learning algorithm that combines 
# Q-learning with deep neural networks to learn optimal policies in complex environments.

# In this scenario, Monitor will be used in order to easily generate training logs during the agent's training.
env = Monitor(IntrustionEnv(train_features_scaled, train_labels))

# We will be using a simple multi-layer perceptron (MLP) policy for our DQN agent, which is suitable for environments with 
# continuous observation spaces like ours.
model = DQN('MlpPolicy', env, verbose=1,
            learning_rate=0.001,
            gamma=0.95,
            exploration_fraction=0.3,      # explore for 30% of training
            exploration_final_eps=0.05,    # settle at 5% random actions
            policy_kwargs=dict(net_arch=[256, 256, 128]))

# Train the DQN agent on the environment for a total of 500,000 timesteps, 
# which should be sufficient for the agent to learn to classify normal traffic and attacks effectively.
print("Training the DQN agent...")
model.set_random_seed(SEED)
model.learn(total_timesteps=500000)

# We save the agent so that we can load it for lated evaluation and testing without having to retrain it from scratch.
model.save("generated/ids_dqn_agent")

# Save the scaler so the real-time inference script uses identical normalization.
joblib.dump(scaler, 'generated/scaler_ids.pkl')
print("Saved: generated/scaler_ids.pkl")

# === EVALUATION === 

# After training the agent, we evaluate its performance on the held-out test set to see how well it has learned to classify normal traffic and attacks.

# In case we do not want to train the agent from scratch, we can load the previously saved model to perform evaluation on the test set.
# model = DQN.load("generated/ids_dqn_agent")

agent_predictions, _states = model.predict(test_features_scaled, deterministic=True)

# After we have collected the agent's predictions for all instances in the test set,
# we can generate a classification report to evaluate the performance of the agent.
report = classification_report(test_labels, agent_predictions, target_names=['Normal Traffic', 'Attack'])
print(report)