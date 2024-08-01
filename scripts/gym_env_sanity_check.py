import gymnasium as gym
import safe_autonomy_sims.gym


def main():
    envs = ["Docking-v0", "Inspection-v0", "WeightedInspection-v0", "SixDofInspection-v0"]

    for env in envs:
        print(f"Testing {env}...")
        env = gym.make(env)
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        env.close()
        print(f"{env} passed!")
    print("All envs passed!")


if __name__ == "__main__":
    main()
