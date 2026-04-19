import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
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

# Split into train/test before scaling to avoid data leakage.
# stratify=answer ensures both splits have the same class distribution.
X_train, X_test, y_train, y_test = train_test_split(
    inputData.values, answer, test_size=0.2, random_state=42, stratify=answer
)

# We need to scale the input data to ensure that all features are on the same scale, which can help the AI agent learn more effectively.
# The scaler is fitted only on the training set; the test set is transformed without fitting to prevent leakage.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
scaler.feature_names_in_ = None  # suppress feature-name warnings during real-time inference

# Print the distribution of attack types and the range of the scaled input data to verify that the preprocessing steps have been applied correctly.
# The first print statement will show us how many instances of normal traffic and different attack types are present in the dataset,
# While the second print statement will confirm that the input data has been scaled to a range between 0 and 1.
print(data.loc[:, 'Attack Type'].value_counts())
print(X_train_scaled.min(), X_train_scaled.max())

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
env = Monitor(IntrustionEnv(X_train_scaled, y_train))

# We will be using a simple multi-layer perceptron (MLP) policy for our DQN agent, which is suitable for environments with 
# continuous observation spaces like ours.
model = DQN('MlpPolicy', env, verbose=1,
            learning_rate=0.001,
            gamma=0.95,
            exploration_fraction=0.3,      # explore for 30% of training
            exploration_final_eps=0.05,    # settle at 5% random actions
            policy_kwargs=dict(net_arch=[256, 256, 128]))
 
# Creăm o instanță separată a mediului doar pentru evaluările din timpul antrenamentului
eval_environment = IntrustionEnv(X_train_scaled, y_train)

# Inițializăm callback-ul (va face o evaluare la fiecare 1000 de pași)
metrics_callback = TrainingMetricsCallback(eval_env=eval_environment, eval_freq=1000, verbose=1)

# The number of steps is ajustable, but we will start with 20,000 steps to allow the agent to learn effectively without taking too long to train.
# In the final thesis, the number of steps will be increased to 500,000.
print("Training the DQN agent...")
model.set_random_seed(SEED)
model.learn(total_timesteps=500000, callback=metrics_callback)

# We save the agent so that we can load it for lated evaluation and testing without having to retrain it from scratch.
model.save("generated/ids_dqn_agent")

# Save the scaler so the real-time inference script uses identical normalization.
joblib.dump(scaler, 'generated/scaler_ids.pkl')
print("Saved: generated/scaler_ids.pkl")

# === EVALUATION === 

# After training the agent, we can evaluate its performance on a test set to see how well it has learned to classify normal traffic and attacks.
# We will use the same CICIDS2017 dataset for evaluation, but we will split it
# into a training set and a test set to ensure that the agent is evaluated on unseen data.

# In case we do not want to train the agent from scratch, we can load the previously saved model and scaler to perform evaluation on the test set.
# model = DQN.load("generated/ids_dqn_agent")

# True labels and predictions use the held-out test set only
true_labels = y_test

agent_predictions, _states = model.predict(X_test_scaled, deterministic=True)

# After we have collected the agent's predictions for all instances in the test set,
# we can generate a classification report to evaluate the performance of the agent.
report = classification_report(true_labels, agent_predictions, target_names=['Normal Traffic', 'Attack'])
print(report)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# REAL VALUES FROM TRAINING
# ============================================================
# Training curves — convert lists to numpy arrays for math operations
steps   = np.array(metrics_callback.steps)
raw_acc = np.array(metrics_callback.accuracies)
raw_rew = np.array(metrics_callback.rewards)

# Confusion matrix — directly from sklearn
conf_matrix = confusion_matrix(true_labels, agent_predictions)

# Per-class metrics — output_dict=True lets us index into the report
report_dict = classification_report(true_labels, agent_predictions,
                                    target_names=['Normal Traffic', 'Attack'],
                                    output_dict=True)
metrics = {
    'Normal Traffic': {
        'Precision': report_dict['Normal Traffic']['precision'],
        'Recall':    report_dict['Normal Traffic']['recall'],
        'F1':        report_dict['Normal Traffic']['f1-score'],
    },
    'Attack': {
        'Precision': report_dict['Attack']['precision'],
        'Recall':    report_dict['Attack']['recall'],
        'F1':        report_dict['Attack']['f1-score'],
    },
}

# ============================================================
# PLOT CONFIGURATION
# ============================================================

BLUE   = '#185FA5'
GREEN  = '#1D9E75'
CORAL  = '#D85A30'
AMBER  = '#BA7517'
PURPLE = '#534AB7'
GRAY   = '#888780'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F8F8',
})

fig = plt.figure(figsize=(16, 12))
fig.suptitle('DQN Intrusion Detection System — Training & Evaluation Report',
             fontsize=16, fontweight='bold', y=0.98)

# 2 rows: top row is full-width accuracy curve, bottom row is reward + confusion matrix
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

window = 25

# ============================================================
# 1. TRAINING ACCURACY CURVE (full width, top row)
# ============================================================
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(steps / 1000, raw_acc * 100, color=BLUE, linewidth=2, label='Accuracy per eval')

smoothed_acc = np.convolve(raw_acc, np.ones(window) / window, mode='valid')
smooth_steps = steps[window - 1:] / 1000
ax1.plot(smooth_steps, smoothed_acc * 100, color=GREEN, linewidth=2.5,
         linestyle='--', label=f'Smoothed (window={window})')

ax1.fill_between(steps / 1000, raw_acc * 100, alpha=0.1, color=BLUE)
ax1.set_title('Training Evolution — Accuracy over Timesteps', fontweight='bold', pad=10)
ax1.set_xlabel('Timesteps (thousands)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(40, 102)
ax1.legend(loc='lower right', framealpha=0.9)

final_acc = raw_acc[-1] * 100
ax1.annotate(f'Final: {final_acc:.1f}%',
             xy=(steps[-1] / 1000, final_acc),
             xytext=(-40, -20), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5),
             fontsize=9, color=BLUE)

# ============================================================
# 2. REWARD CURVE (bottom left)
# ============================================================
ax2 = fig.add_subplot(gs[1, 0])

ax2.plot(steps / 1000, raw_rew, color=GREEN, linewidth=2, alpha=0.6, label='Episode reward')
smoothed_rew = np.convolve(raw_rew, np.ones(window) / window, mode='valid')
ax2.plot(steps[window - 1:] / 1000, smoothed_rew, color='#085041',
         linewidth=2.5, linestyle='--', label='Smoothed')
ax2.fill_between(steps / 1000, raw_rew, alpha=0.12, color=GREEN)
ax2.axhline(0, color='black', linewidth=0.7, alpha=0.4, linestyle=':')
ax2.set_title('Reward per Episode during Training', fontweight='bold', pad=10)
ax2.set_xlabel('Timesteps (thousands)')
ax2.set_ylabel('Total reward')
ax2.legend(loc='lower right', framealpha=0.9, fontsize=9)

# ============================================================
# 3. CONFUSION MATRIX (bottom right)
# ============================================================
ax3 = fig.add_subplot(gs[1, 1])

cmap = LinearSegmentedColormap.from_list('blue_white', ['#E6F1FB', '#185FA5'])
im = ax3.imshow(conf_matrix, cmap=cmap, aspect='auto')

tick_labels = ['Normal\nTraffic', 'Attack']
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(tick_labels, fontsize=10)
ax3.set_yticklabels(tick_labels, fontsize=10)
ax3.set_xlabel('Predicted label', fontsize=11)
ax3.set_ylabel('True label', fontsize=11)
ax3.set_title('Confusion Matrix', fontweight='bold', pad=10)
ax3.spines[:].set_visible(False)

total = conf_matrix.sum()
for i in range(2):
    for j in range(2):
        val = conf_matrix[i, j]
        pct = val / total * 100
        text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else '#0C447C'
        ax3.text(j, i, f'{val:,}\n({pct:.1f}%)',
                 ha='center', va='center', fontsize=11,
                 fontweight='bold', color=text_color)

cell_labels = [['TN', 'FP'], ['FN', 'TP']]
label_colors = [['#1D9E75', '#D85A30'], ['#D85A30', '#1D9E75']]
for i in range(2):
    for j in range(2):
        ax3.text(j, i - 0.35, cell_labels[i][j],
                 ha='center', va='center', fontsize=9,
                 color=label_colors[i][j], fontweight='bold')

plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# ============================================================
# SAVE
# ============================================================
plt.savefig('ids_dqn_report.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: ids_dqn_report.png")
plt.show()


# ============================================================
# SEPARATE FIGURE: Per-class Precision / Recall / F1
# ============================================================

fig2, ax4 = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('white')
ax4.set_facecolor('#F8F8F8')

classes = list(metrics.keys())
metric_names = ['Precision', 'Recall', 'F1']
metric_colors = [BLUE, GREEN, AMBER]
x = np.arange(len(classes))
width = 0.25

for i, (metric_name, color) in enumerate(zip(metric_names, metric_colors)):
    vals = [metrics[cls][metric_name] * 100 for cls in classes]
    bars = ax4.bar(x + i * width, vals, width, label=metric_name,
                   color=color, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=GRAY)

ax4.set_title('Per-class Precision, Recall & F1', fontweight='bold', pad=10)
ax4.set_ylabel('Score (%)')
ax4.set_ylim(80, 102)
ax4.set_xticks(x + width)
ax4.set_xticklabels(classes, fontsize=11)
ax4.legend(framealpha=0.9)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(axis='y', alpha=0.25, linestyle='--')

plt.tight_layout()
plt.savefig('ids_dqn_metrics.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: ids_dqn_metrics.png")
plt.show()