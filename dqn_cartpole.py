import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment using Gymnasium
env = gym.make("CartPole-v1")

# Initialize and train the model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# Test the model
obs = env.reset()[0]
print(obs)
for _ in range(1000):
    action, _ = model.predict(obs)
    print(action)
    obs, reward, done, info, extra_value = env.step(action)  # For 5 returned values
    
    env.render()
    if done:
        obs = env.reset()[0]
    #print(obs)
env.close()
