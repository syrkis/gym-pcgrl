import gymnasium as gym
from gymnasium import envs
import pcgymrl
import matplotlib.pyplot as plt


def main():
    env = gym.make("zelda-turtle-v0", render_mode="rgb_array")
    obs = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated | truncated:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == "__main__":
    main()
