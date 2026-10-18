"""Check the launch Gym interface against a running KSP/kRPC session."""

from stable_baselines3.common.env_checker import check_env

if __package__:
    from .orbit_command import KSPCommandedOrbitalEnv
else:
    from orbit_command import KSPCommandedOrbitalEnv


def main():
    env = KSPCommandedOrbitalEnv()
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
