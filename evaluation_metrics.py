import streamlit as st
import gymnasium as gym
import numpy as np 
import time
import pyautogui
from stable_baselines3 import DQN

# Tracking the performance

env = gym.make("CartPole-v1")
model = DQN.load("models/dqn_cartpole")




total_rewards = 0
steps = 0
obs = env.reset()[0]


done = False
while not done:
    # Predict the action
    action, _ = model.predict(obs)

    # Take the action
    obs, reward, done, info, _ = env.step(action)

    # Accumulate rewards
    total_rewards += reward
    steps += 1

print(f"Total reward: {total_rewards}, Steps: {steps}")
