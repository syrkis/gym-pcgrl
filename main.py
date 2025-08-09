import pcgym


def main():
    env = pcgym.make("smb-turtle-v0", render_mode="rgb_array")
    obs = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated | truncated:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == "__main__":
    main()
