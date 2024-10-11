from typing import Dict

import gym
import numpy as np

from jaxrl2.data.dataset import Dataset


def evaluate(agent, env: gym.Env, num_episodes: int, random_agent:bool=False) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            if random_agent:
                action = env.action_space.sample()
            else:
                action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}


def evaluate_log_prob(agent, dataset: Dataset, batch_size: int = 2048) -> float:
    num_iters = len(dataset) // batch_size
    total_log_prob = 0.0
    for j in range(num_iters):
        indx = np.arange(j * batch_size, (j + 1) * batch_size)
        batch = dataset.sample(batch_size, keys=("observations", "actions"), indx=indx)
        log_prob = agent.eval_log_probs(batch)
        total_log_prob += log_prob

    return total_log_prob / num_iters


#  for evaluating the distance between the learning the Q function and the policy

import math

def estimate_true_values(trajectories, gamma=0.99):
    """
    Estimate the true Q-values for each state-action pair in the collected trajectories using Monte Carlo returns.
    
    Parameters:
    - trajectories: A list of episodes, each containing (state, action, reward) tuples.
    - gamma: The discount factor.
    
    Returns:
    - observations: A numpy array of observations.
    - actions: A numpy array of actions.
    - Gs: A numpy array of estimated returns.
    """
    observations = []
    actions = []
    Gs = []

    for episode_trajectory in trajectories:
        G = 0  # Initialize the return
        for t in reversed(range(len(episode_trajectory))):
            observation, action, reward = episode_trajectory[t]
            G = reward + gamma * G  # Discounted return
            observations.append(observation)
            actions.append(action)
            Gs.append(G)
    
    return np.array(observations), np.array(actions), np.array(Gs)

def calculate_distance(agent, observations, actions, Gs):
    """
    Calculate the distance between the Q-function and the estimated true Q-values.
    
    Parameters:
    - agent: the agent to test
    - observations: A numpy array of observations.
    - actions: A numpy array of actions.
    - Gs: A numpy array of estimated returns.
    
    Returns:
    - mse: The root mean squared error between the Q-function and the estimated true Q-values.
    """
    total_samples = len(observations)

    total_squared_error = np.linalg.norm(agent.get_Q_value(observations, actions) - Gs)
    # Calculate Root Mean Squared Error (MSE)
    mse = math.sqrt(total_squared_error) / total_samples if total_samples > 0 else float('inf')
    return mse


def evaluate_with_disc(agent, env: gym.Env, num_episodes: int, random_agent:bool=False) -> Dict[str, float]:
    trajectories = []

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for episode in range(num_episodes):
        observation, done = env.reset(), False
        episode_trajectory = []

        while not done:
            if random_agent:
                action = env.action_space.sample()
            else:
                action = agent.eval_actions(observation)
            observation, reward, done, _ = env.step(action)
            episode_trajectory.append((observation, action, reward))

        trajectories.append(episode_trajectory)

    observations, actions, Gs = estimate_true_values(trajectories, agent.discount)
    rmse = calculate_distance(agent, observations, actions, Gs)

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue), "rmse": rmse}


