import time

from stable_baselines3 import PPO

from gym_training import KSPOrbitalEnv


MODEL_PATH = "./ppo_ksp_brain_v2"


def run_agent(number_of_flights=10):
    print("Connecting to KSP and creating the environment...")
    env = KSPOrbitalEnv()

    print(f"Loading trained pilot from {MODEL_PATH}.zip...")
    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="cpu",
    )

    successes = 0
    failures = 0
    timeouts = 0

    try:
        for episode in range(1, number_of_flights + 1):
            print("\n" + "=" * 60)
            print(f"STARTING EVALUATION FLIGHT {episode}/{number_of_flights}")
            print("=" * 60)

            observation, info = env.reset()

            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0

            while not terminated and not truncated:
                # deterministic=True disables training exploration.
                # The pilot uses its preferred action instead of sampling
                # a random variation from its policy distribution.
                action, _state = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, reward, terminated, truncated, info = env.step(
                    action
                )

                episode_reward += reward
                episode_steps += 1

            apoapsis = info.get("apoapsis", 0.0)
            periapsis = info.get("periapsis", -600_000.0)

            if terminated and periapsis > env.target_pe:
                successes += 1
                result = "ORBIT ACHIEVED"

            elif truncated:
                timeouts += 1
                result = "TIMEOUT"

            else:
                failures += 1
                result = "FAILED"

            print("\n" + "-" * 60)
            print(f"FLIGHT {episode} RESULT: {result}")
            print(f"Steps:          {episode_steps}")
            print(f"Episode reward: {episode_reward:.1f}")
            print(f"Apoapsis:       {apoapsis / 1000:.1f} km")
            print(f"Periapsis:      {periapsis / 1000:.1f} km")
            print("-" * 60)

            # Brief pause before the next save reload.
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

    finally:
        env.close()

    completed = successes + failures + timeouts

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Completed flights: {completed}")
    print(f"Successful orbits: {successes}")
    print(f"Failures:          {failures}")
    print(f"Timeouts:          {timeouts}")

    if completed > 0:
        success_rate = 100.0 * successes / completed
        print(f"Orbit success rate: {success_rate:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    run_agent(number_of_flights=10)