from collections import Counter

from stable_baselines3 import PPO

from orbit_command import KSPCommandedOrbitalEnv


MODEL_PATH = "./ppo_ksp_commanded_pilot_v4_latest"


EVAL_COMMANDS = [
    {
        "target_ap": 90_000.0,
        "target_pe": 75_000.0,
        "min_fuel_fraction": 0.02,
        "urgency": 0.5,
    },
    {
        "target_ap": 100_000.0,
        "target_pe": 80_000.0,
        "min_fuel_fraction": 0.03,
        "urgency": 0.5,
    },
    {
        "target_ap": 120_000.0,
        "target_pe": 90_000.0,
        "min_fuel_fraction": 0.03,
        "urgency": 0.5,
    },
    {
        "target_ap": 130_000.0,
        "target_pe": 100_000.0,
        "min_fuel_fraction": 0.03,
        "urgency": 0.5,
    },
    {
        "target_ap": 150_000.0,
        "target_pe": 120_000.0,
        "min_fuel_fraction": 0.02,
        "urgency": 0.5,
    },
    {
        "target_ap": 170_000.0,
        "target_pe": 130_000.0,
        "min_fuel_fraction": 0.01,
        "urgency": 0.5,
    },
]


def run_eval_episode(model, env, command, episode_index):
    print("\n" + "=" * 70)
    print(f"DETERMINISTIC EVALUATION EPISODE {episode_index}")
    print(
        f"Command: "
        f"Ap {command['target_ap'] / 1000:.1f} km | "
        f"Pe {command['target_pe'] / 1000:.1f} km | "
        f"Min fuel {command['min_fuel_fraction']:.2f} | "
        f"Urgency {command['urgency']:.2f}"
    )
    print("=" * 70)

    obs, info = env.reset(options=command)

    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    target_ap = info["command_target_ap"]
    target_pe = info["command_target_pe"]
    actual_ap = info["apoapsis"]
    actual_pe = info["periapsis"]

    ap_error = abs(actual_ap - target_ap)
    pe_error = abs(actual_pe - target_pe)

    fuel_margin = (
        info["fuel_fraction"]
        - info["command_min_fuel_fraction"]
    )

    result = {
        "episode": episode_index,
        "success": bool(info["is_success"]),
        "reason": info["termination_reason"],
        "steps": steps,
        "total_reward": total_reward,

        "target_ap_km": target_ap / 1000.0,
        "actual_ap_km": actual_ap / 1000.0,
        "ap_error_km": ap_error / 1000.0,
        "ap_error_tolerances": info["normalized_apoapsis_error"],

        "target_pe_km": target_pe / 1000.0,
        "actual_pe_km": actual_pe / 1000.0,
        "pe_error_km": pe_error / 1000.0,
        "pe_error_tolerances": info["normalized_periapsis_error"],

        "fuel_fraction": info["fuel_fraction"],
        "required_fuel": info["command_min_fuel_fraction"],
        "fuel_margin": fuel_margin,
    }

    print("\nEPISODE RESULT")
    print(f"Success:        {result['success']}")
    print(f"Reason:         {result['reason']}")
    print(f"Steps:          {result['steps']}")
    print(f"Total reward:   {result['total_reward']:.2f}")

    print(
        f"Ap:             "
        f"{result['actual_ap_km']:.1f} / "
        f"{result['target_ap_km']:.1f} km | "
        f"error {result['ap_error_km']:.1f} km | "
        f"{result['ap_error_tolerances']:.2f} tolerances"
    )

    print(
        f"Pe:             "
        f"{result['actual_pe_km']:.1f} / "
        f"{result['target_pe_km']:.1f} km | "
        f"error {result['pe_error_km']:.1f} km | "
        f"{result['pe_error_tolerances']:.2f} tolerances"
    )

    print(
        f"Fuel:           "
        f"{result['fuel_fraction']:.3f} | "
        f"required {result['required_fuel']:.3f} | "
        f"margin {result['fuel_margin']:+.3f}"
    )

    return result


def main():
    env = KSPCommandedOrbitalEnv(
        randomize_commands=False,
    )

    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="cpu",
    )

    results = []

    for index, command in enumerate(EVAL_COMMANDS, start=1):
        result = run_eval_episode(
            model=model,
            env=env,
            command=command,
            episode_index=index,
        )

        results.append(result)

    env.close()

    total = len(results)
    successes = sum(result["success"] for result in results)
    success_rate = successes / total

    reasons = Counter(result["reason"] for result in results)

    avg_ap_error = (
        sum(result["ap_error_km"] for result in results)
        / total
    )

    avg_pe_error = (
        sum(result["pe_error_km"] for result in results)
        / total
    )

    avg_fuel_margin = (
        sum(result["fuel_margin"] for result in results)
        / total
    )

    print("\n" + "=" * 70)
    print("DETERMINISTIC EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Episodes:        {total}")
    print(f"Successes:       {successes}")
    print(f"Success rate:    {success_rate * 100:.1f}%")
    print(f"Avg Ap error:    {avg_ap_error:.1f} km")
    print(f"Avg Pe error:    {avg_pe_error:.1f} km")
    print(f"Avg fuel margin: {avg_fuel_margin:+.3f}")

    print("\nTermination reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    print("\nPer-episode compact summary:")
    for result in results:
        status = "SUCCESS" if result["success"] else "FAIL"

        print(
            f"#{result['episode']} | "
            f"{status} | "
            f"{result['reason']} | "
            f"Ap err {result['ap_error_km']:.1f} km | "
            f"Pe err {result['pe_error_km']:.1f} km | "
            f"Fuel margin {result['fuel_margin']:+.3f}"
        )


if __name__ == "__main__":
    main()