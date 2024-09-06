import streamlit as st
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

# Set up Streamlit UI
st.title('CartPole AI Playing - Deep Q-Learning Visualization')

# Placeholder for rendering game frames
frame_placeholder = st.empty()  # To show frames of the game
info_placeholder = st.empty()   # To show episode information

# Load the environment and model
env = gym.make('CartPole-v1')
model = DQN.load('models/dqn_cartpole')

# Reset environment
obs = env.reset()[0]
done = False

# Variables to track total reward and steps
total_reward = 0
steps = 0

# Create a placeholder plot
fig, ax = plt.subplots()
ax.axis('off')  # Hide axes

# Load a placeholder image (you can use a sample CartPole image here)
placeholder_image = np.ones((400, 600, 3))  # Placeholder for CartPole image

def render_cartpole():
    # Use a placeholder image for visualization
    ax.clear()
    ax.imshow(placeholder_image)  # Display placeholder image
    ax.axis('off')

# Run the AI in the environment and render it in Streamlit
while not done:
    # Predict action using the trained model
    action, _ = model.predict(obs)
    
    # Take the predicted action in the environment
    obs, reward, done, info,_ = env.step(action)
    
    # Increment the total reward and steps
    total_reward += reward
    steps += 1
    
    # Render the current frame of the environment (placeholder)
    render_cartpole()
    
    # Convert Matplotlib plot to a NumPy array and display on Streamlit
    frame_placeholder.pyplot(fig)
    
    # Update the information text about the total reward and steps
    info_placeholder.text(f'Total Reward: {total_reward}, Steps: {steps}')
    
    # Add a small delay to slow down the visualization
    time.sleep(0.03)

# Close the environment once done
env.close()

# Display final result after the episode ends
st.write(f"Episode finished. Total reward: {total_reward}, Total steps: {steps}")
