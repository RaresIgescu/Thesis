import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib

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
            reward = -1.0
        elif(action == 1 and true_label == 0):
            reward = -2.0

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
        max_limit = len(self.features) - self.maxSteps - 1
        self.current_step = np.random.randint(0, max_limit)
        self.steps_taken = 0
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

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        
        # Aici vom stoca datele pentru grafic
        self.steps = []
        self.rewards = []
        self.accuracies = []

    def _on_step(self) -> bool:
        # Evaluăm modelul la fiecare 'eval_freq' pași
        if self.n_calls % self.eval_freq == 0:
            # Rulăm un episod de testare pentru a vedea performanța curentă
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            correct_predictions = 0
            total_steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                total_steps += 1
                
                # În mediul tău, dacă primește +1 înseamnă că a ghicit corect
                if reward == 1.0:
                    correct_predictions += 1
                    
            accuracy = correct_predictions / total_steps if total_steps > 0 else 0
            
            # Salvăm metricile
            self.steps.append(self.num_timesteps)
            self.rewards.append(total_reward)
            self.accuracies.append(accuracy)
            
            if self.verbose > 0:
                print(f"Step: {self.num_timesteps} | Reward: {total_reward} | Accuracy: {accuracy:.2f}")
                
        return True

# === DEFINE THE AGENT ===

# Considering that we have defined the environment in which the agent will be trained, we can now define the agent itself.
# We will be using the DQN algorithm from the stable_baselines3 library, which is a popular reinforcement learning algorithm that combines 
# Q-learning with deep neural networks to learn optimal policies in complex environments.

# In this scenario, Monitor will be used in order to easily generate training logs and visualize the agent's performance over time.
env = Monitor(IntrustionEnv(inputData_scaled, answer))

# We will be using a simple multi-layer perceptron (MLP) policy for our DQN agent, which is suitable for environments with 
# continuous observation spaces like ours.
model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001, gamma=0.0)

# Creăm o instanță separată a mediului doar pentru evaluările din timpul antrenamentului
eval_environment = IntrustionEnv(inputData_scaled, answer)

# Inițializăm callback-ul (va face o evaluare la fiecare 1000 de pași)
metrics_callback = TrainingMetricsCallback(eval_env=eval_environment, eval_freq=1000, verbose=1)

# The number of steps is ajustable, but we will start with 20,000 steps to allow the agent to learn effectively without taking too long to train.
# In the final thesis, the number of steps will be increased to 500,000.
print("Training the DQN agent...")
model.learn(total_timesteps=500000, callback=metrics_callback)

# We save the agent so that we can load it for lated evaluation and testing without having to retrain it from scratch.
model.save("generated/ids_dqn_agent")

# Consdering that I will integrate the AI itself in a real-time protection sysytem, once I analyze a HTTP request,
# the data will be preprocessed in the same way as the training data and then fed into the trained DQN agent 
# to get a prediction on whether the traffic is normal or an attack.
joblib.dump(scaler, 'generated/scaler_ids.pkl')

# === EVALUATION === 

# After training the agent, we can evaluate its performance on a test set to see how well it has learned to classify normal traffic and attacks.
# We will use the same CICIDS2017 dataset for evaluation, but we will split it
# into a training set and a test set to ensure that the agent is evaluated on unseen data.

# In case we do not want to train the agent from scratch, we can load the previously saved model and scaler to perform evaluation on the test set.
# model = DQN.load("generated/ids_dqn_agent")

# True labels, 0 for Normal and 1 for Attack
true_labels = answer

# Here we will save the agents predictions
agent_predictions = []

# Trecem TOATE datele deodată prin model (vectorizat). 
# PyTorch va calcula predicțiile pentru toate pachetele simultan!

agent_predictions, _states = model.predict(inputData_scaled, deterministic=True) 

# After we have collected the agent's predictions for all instances in the test set,
# we can generate a classification report to evaluate the performance of the agent.
report = classification_report(true_labels, agent_predictions, target_names=['Normal Traffic', 'Attack'])
print(report)

confusionMatrix = confusion_matrix(true_labels, agent_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=['Normal Traffic', 'Attack'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Matricea de Confuzie - DQN IDS")

plt.savefig("generated/confusion_matrix.png", dpi=300, bbox_inches='tight')

plt.show()

# === Generarea Graficului de Evoluție ===

fig, ax1 = plt.subplots(figsize=(10, 6))

# Axa Y din stânga - pentru Recompensă
color = 'tab:blue'
ax1.set_xlabel('Pași de Antrenament')
ax1.set_ylabel('Recompensă pe Episod', color=color)
ax1.plot(metrics_callback.steps, metrics_callback.rewards, color=color, linewidth=2, marker='o', label='Recompensă')
ax1.tick_params(axis='y', labelcolor=color)

# Axa Y din dreapta - pentru Acuratețe
ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel('Acuratețe', color=color)
ax2.plot(metrics_callback.steps, metrics_callback.accuracies, color=color, linewidth=2, marker='s', label='Acuratețe')
ax2.tick_params(axis='y', labelcolor=color)

# Adăugăm un titlu și combinăm legendele
plt.title('Evoluția Recompensei și Acurateței în timpul Antrenamentului')
fig.tight_layout()

# Punem legenda sus în stânga
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Salvăm graficul
plt.savefig("generated/training_evolution.png", dpi=300, bbox_inches='tight')
print("Graficul evoluției a fost salvat în 'generated/training_evolution.png'.")

plt.show()