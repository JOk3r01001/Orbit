"""Offline landing checks. Transport/Gym tests use no KSP connection."""

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mission_control.commands import LandingTarget
from mission_control.landing import LandingSample, distance_to_target, touchdown_failure, transition_reward

try:
    import numpy as np
    from scripts.landing_train import LandingEnv
except ImportError:
    LandingEnv = None


def sample(**changes):
    return replace(LandingSample(100, 400, -15, .1, 0, 1, 25, 9.81, 0, 0, "flying"), **changes)


class LandingLogicTests(unittest.TestCase):
    def test_zero_reported_landed_velocity_does_not_hide_hard_impact(self):
        self.assertEqual(touchdown_failure(sample(vertical_mps=-25), sample(situation="landed", vertical_mps=0), LandingTarget(0, 0)), "hard_touchdown")

    def test_soft_contact_within_goal_is_accepted(self):
        self.assertIsNone(touchdown_failure(sample(vertical_mps=-1), sample(situation="landed", vertical_mps=0), LandingTarget(0, 0)))

    def test_damaged_or_tipped_or_distant_contact_is_rejected(self):
        before = sample(vertical_mps=-1)
        for changes, reason in (({"intact": False}, "vessel_damaged"), ({"tilt_deg": 15}, "touchdown_tilt"), ({"latitude_deg": 1}, "outside_landing_site")):
            with self.subTest(reason=reason):
                self.assertEqual(touchdown_failure(before, sample(situation="landed", vertical_mps=0, **changes), LandingTarget(0, 0)), reason)

    def test_hovering_is_not_rewarded(self):
        before = sample(vertical_mps=0)
        reward, _ = transition_reward(before, replace(before, ut_s=100.1), LandingTarget(0, 0))
        self.assertLess(reward, 0)

    def test_fuel_usage_has_an_explicit_cost(self):
        before = sample()
        _, terms = transition_reward(before, replace(before, ut_s=100.1, fuel_fraction=.9), LandingTarget(0, 0))
        self.assertAlmostEqual(terms["fuel_penalty"], .5)

    def test_nonfinite_and_stale_telemetry_rejected(self):
        with self.assertRaises(ValueError):
            sample(height_m=float("nan"))
        with self.assertRaises(ValueError):
            transition_reward(sample(), sample(), LandingTarget(0, 0))

    def test_longitude_wrap_distance(self):
        self.assertLess(distance_to_target(sample(longitude_deg=179.999), LandingTarget(0, -179.999)), 30)


@unittest.skipIf(LandingEnv is None, "Optional training dependencies are not installed")
class LandingEnvironmentTests(unittest.TestCase):
    def resettable_env(self):
        body = SimpleNamespace(name="Kerbin")
        vessel = SimpleNamespace(
            control=SimpleNamespace(throttle=0), auto_pilot=Mock(), surface_reference_frame=object(),
            orbit=SimpleNamespace(body=body),
            resources=SimpleNamespace(amount=lambda _: 100),
            parts=SimpleNamespace(all=[1, 2, 3], parachutes=[]),
        )
        connection = SimpleNamespace(space_center=SimpleNamespace(active_vessel=vessel, load=Mock()), close=Mock())
        env = LandingEnv(connection=connection)
        self.ut = 100

        def read():
            self.ut += .2
            return sample(ut_s=self.ut)

        env.read = Mock(side_effect=read)
        return env

    def env(self, previous=None):
        env = LandingEnv.__new__(LandingEnv)
        env.previous = previous or sample()
        env.target = LandingTarget(0, 0)
        env.vessel = SimpleNamespace(control=SimpleNamespace(throttle=0))
        env.stop = Mock()
        env.read = Mock()
        env.timeout, env.settle_time, env.reserve = 180, .2, 0
        env.steps, env.throttle, env.started_ut = 0, 0, 100
        env.contact, env.done = None, False
        return env

    def step(self, env, next_sample, action=.5):
        env.read.return_value = next_sample
        with patch("scripts.landing_train.time.sleep"):
            return env.step(np.array([action], dtype=np.float32))

    def test_soft_contact_requires_engines_off_settling(self):
        env = self.env(sample(vertical_mps=-1, height_m=3))
        _, _, terminated, _, info = self.step(env, sample(ut_s=100.2, vertical_mps=0, height_m=3, situation="landed"))
        self.assertFalse(terminated)
        self.assertEqual(env.vessel.control.throttle, 0)
        obs, _, terminated, _, info = self.step(env, sample(ut_s=100.5, vertical_mps=0, height_m=3, situation="landed"))
        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertEqual(obs.shape, (14,))
        env.stop.assert_called_once()

    def test_bounce_after_contact_fails(self):
        env = self.env()
        env.contact = (100, 1, .1)
        _, _, terminated, _, info = self.step(env, sample(ut_s=100.2))
        self.assertTrue(terminated)
        self.assertEqual(info["reason"], "bounced_after_touchdown")

    def test_hard_contact_fails_even_with_zero_landed_speed(self):
        env = self.env()
        _, reward, terminated, _, info = self.step(env, sample(ut_s=100.2, situation="landed", vertical_mps=0))
        self.assertTrue(terminated)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["reason"], "hard_touchdown")
        self.assertLess(reward, 0)

    def test_alignment_blocks_policy_throttle(self):
        env = self.env(sample(tilt_deg=20))
        _, _, _, _, info = self.step(env, sample(ut_s=100.2, tilt_deg=20), 1)
        self.assertEqual(env.vessel.control.throttle, 0)
        self.assertTrue(info["alignment_intervention"])

    def test_telemetry_failure_requests_cutoff(self):
        env = self.env()
        env.read.side_effect = ConnectionError("disconnected")
        with self.assertRaises(ConnectionError):
            self.step(env, sample(ut_s=100.2))
        env.stop.assert_called_once()

    def test_invalid_policy_action_requests_cutoff(self):
        env = self.env()
        with self.assertRaises(ValueError):
            self.step(env, sample(ut_s=100.2), float("nan"))
        env.stop.assert_called_once()

    def test_confirmed_destroyed_vessel_ends_episode_for_reload(self):
        env = self.env()
        env.sc = SimpleNamespace(vessels=[])
        env.read.side_effect = RuntimeError("Remote vessel no longer exists")
        _, reward, terminated, truncated, info = self.step(env, sample(ut_s=100.2))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -100)
        self.assertEqual(info["reason"], "vessel_destroyed")
        self.assertIsNone(env.vessel)

    def test_surviving_switched_vessel_is_not_treated_as_crash(self):
        env = self.env()
        env.sc = SimpleNamespace(vessels=[env.vessel])
        env.read.side_effect = RuntimeError("Active vessel changed")
        with self.assertRaisesRegex(RuntimeError, "Active vessel changed"):
            self.step(env, sample(ut_s=100.2))
        env.stop.assert_called_once()

    def test_timeout_is_a_failed_truncation(self):
        env = self.env()
        env.timeout = .1
        _, _, terminated, truncated, info = self.step(env, sample(ut_s=100.2))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "timeout")

    def test_fuel_reserve_applies_even_at_successful_contact(self):
        env = self.env(sample(vertical_mps=-1))
        env.contact = (99, 1, .1)
        env.reserve = .1
        _, _, _, _, info = self.step(env, sample(ut_s=100.2, situation="landed", vertical_mps=0, fuel_fraction=.05))
        self.assertFalse(info["is_success"])
        self.assertEqual(info["reason"], "fuel_reserve_violated")

    def test_reset_loads_descent_save_and_holds_upright_without_staging(self):
        env = self.resettable_env()
        with patch("scripts.landing_train.time.sleep"):
            obs, _ = env.reset(seed=1)
        env.sc.load.assert_called_once_with("landing_training_start")
        env.vessel.auto_pilot.target_pitch_and_heading.assert_called_once_with(90, 90)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.vessel.control.throttle, 0)
        env.close()

    def test_reset_rejects_unrecoverable_descent(self):
        env = self.resettable_env()
        env.read.side_effect = None
        env.read.return_value = sample(height_m=100, vertical_mps=-40, acceleration_mps2=13)
        with patch("scripts.landing_train.time.sleep"), self.assertRaisesRegex(ValueError, "too fast/low"):
            env.reset()
        self.assertEqual(env.vessel.control.throttle, 0)
        self.assertFalse(env.vessel.auto_pilot.engaged)

    def test_real_gym_and_ppo_checkpoint_resume_with_fake_telemetry(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        env = self.resettable_env()
        with patch("scripts.landing_train.time.sleep"), tempfile.TemporaryDirectory() as directory:
            check_env(env, warn=True, skip_render_check=True)
            model = PPO("MlpPolicy", env, n_steps=8, batch_size=4, n_epochs=1,
                        policy_kwargs={"net_arch": [8]}, device="cpu", seed=7)
            model.learn(total_timesteps=8)
            checkpoint = Path(directory) / "landing.zip"
            model.save(str(checkpoint))
            resumed = PPO.load(str(checkpoint), env=env, device="cpu")
            self.assertEqual(resumed.num_timesteps, 8)
            resumed.learn(total_timesteps=8, reset_num_timesteps=False)
            self.assertEqual(resumed.num_timesteps, 16)
        env.close()


if __name__ == "__main__":
    unittest.main()
