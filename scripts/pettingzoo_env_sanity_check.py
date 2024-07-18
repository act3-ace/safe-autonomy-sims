import pettingzoo.test as test
import safe_autonomy_sims.pettingzoo as sa_zoo


def main():
    envs = [sa_zoo.MultiDockingEnv]

    for env in envs:
        print(f"Testing {env}...")
        test.parallel_api_test(env(), num_cycles=100)
        print(f"{env} passed!")
    print("All envs passed!")


if __name__ == "__main__":
    main()
