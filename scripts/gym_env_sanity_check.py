import tqdm
import gymnasium as gym
import safe_autonomy_sims.gym


def main():
    envs = ["Docking-v0", "Inspection-v0", "WeightedInspection-v0", "SixDofInspection-v0"]

    for env in envs:
        print(f"Testing {env}...")
        env = gym.make(env)
        env.reset()
        for i in tqdm.tqdm(range(100)):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Terminated. Resetting...")
                env.reset()
        env.close()
        print(f"{env} passed!")
    print("All envs passed!")


if __name__ == "__main__":
    main()
