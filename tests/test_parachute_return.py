"""Exercise the default return path without KSP or a PPO checkpoint."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mission_control.commands import DeorbitTarget, MissionTask, TaskType
from mission_control.krpc_adapter import KRPCOrbitalAdapter
from mission_control.recovery import ParachuteRecovery, ParachuteReturnMission, RecoverySample
from scripts.return_to_ground import main, run_parachute_return


def sample(ut, height, vertical, state="stowed", situation="flying", intact=True):
    return RecoverySample(ut, situation, height, vertical, 0, 0, 0, intact, (state,))


class ParachuteReturnTests(unittest.TestCase):
    def setUp(self):
        self.adapter = Mock(spec=KRPCOrbitalAdapter)
        self.chute = Mock(state=SimpleNamespace(name="stowed"))
        self.adapter.vessel = SimpleNamespace(parts=SimpleNamespace(all=[1, 2], parachutes=[self.chute]), flight=Mock())
        self.adapter.body = SimpleNamespace(reference_frame=object())
        self.task = MissionTask("return-test", TaskType.DEORBIT, DeorbitTarget())

    def run_mission(self, samples, config=ParachuteRecovery(settle_time_s=1), deploy=(True,)):
        self.adapter.read_recovery.side_effect = samples
        self.adapter.deploy_parachute_if_safe.side_effect = deploy
        with tempfile.TemporaryDirectory() as directory, patch("scripts.return_to_ground.time.sleep"), patch(
            "scripts.return_to_ground.run_live", return_value={"result": {"status": "succeeded"}}
        ) as burn:
            result = run_parachute_return(ParachuteReturnMission(self.task, config), self.adapter, Path(directory))
            self.assertTrue(list(Path(directory).glob("*.json")))
            self.burn_count = burn.call_count
            return result

    def test_unsafe_entry_waits_and_full_canopy_landing_succeeds(self):
        report = self.run_mission([
            sample(100, 12_000, -500), sample(101, 9000, -450), sample(102, 5000, -150),
            sample(103, 3000, -100, "stowed"), sample(104, 1000, -30, "semi_deployed"),
            sample(105, 5, -6, "deployed"), sample(106, 0, 0, "cut", "landed"),
            sample(107, 0, 0, "cut", "landed"),
        ], deploy=(False, True))
        self.assertEqual(report["status"], "succeeded", report.get("reason"))
        self.assertEqual(report["deploy_commands"], [{"index": 0, "ut_s": 102, "height_m": 5000}])
        self.assertEqual(self.adapter.deploy_parachute_if_safe.call_count, 2)
        self.assertEqual(self.adapter.apply.call_count, 2)
        self.assertTrue(all(call.args[0].throttle == 0 for call in self.adapter.apply.call_args_list))
        self.assertEqual(report["touchdown"]["last_airborne"]["vertical_mps"], -6)
        self.assertEqual(self.burn_count, 1)

    def test_missing_chutes_rejected_before_deorbit_or_control(self):
        self.adapter.vessel.parts.parachutes = []
        report = self.run_mission([])
        self.assertEqual(report["status"], "failed")
        self.assertIn("stock parachute", report["reason"])
        self.assertEqual(self.burn_count, 0)
        self.adapter.stop.assert_not_called()

    def test_zero_landed_speed_does_not_hide_fast_touchdown(self):
        result = self.run_mission([
            sample(100, 100, -40), sample(101, 5, -25, "deployed"), sample(102, 0, 0, "cut", "landed"),
        ])
        self.assertEqual(result["status"], "failed")
        self.assertIn("Pre-contact speed", result["reason"])

    def test_landing_without_observed_full_canopy_fails(self):
        result = self.run_mission([
            sample(100, 2, -3, "semi_deployed"), sample(101, 0, 0, "semi_deployed", "landed"),
        ])
        self.assertEqual(result["status"], "failed")
        self.assertIn("fully open parachute", result["reason"])
        self.adapter.deploy_parachute_if_safe.assert_not_called()

    def test_passive_entry_and_splashdown(self):
        report = self.run_mission([
            sample(100, 9000, -250), sample(101, 2, -5, "deployed"),
            sample(102, 0, 0, "cut", "splashed"), sample(103, 0, 0, "cut", "splashed"),
        ], ParachuteRecovery(entry_attitude="passive", settle_time_s=1))
        self.assertEqual(report["status"], "succeeded", report.get("reason"))
        self.assertIn("Splashdown", report["reason"])
        self.adapter.apply.assert_not_called()

    def test_adapter_only_calls_deploy_for_stowed_safe_chute(self):
        adapter = KRPCOrbitalAdapter.__new__(KRPCOrbitalAdapter)
        adapter._assert_bound = Mock()
        for state, safety in [("stowed", "unsafe"), ("stowed", "risky"), ("armed", "safe")]:
            chute = Mock(state=SimpleNamespace(name=state), safe_state=SimpleNamespace(name=safety))
            self.assertFalse(adapter.deploy_parachute_if_safe(chute))
            chute.deploy.assert_not_called()
        self.chute.safe_state = SimpleNamespace(name="safe")
        self.assertTrue(adapter.deploy_parachute_if_safe(self.chute))
        self.chute.deploy.assert_called_once()

    def test_execute_without_model_uses_parachute_runner(self):
        connection = Mock()
        with patch("mission_control.krpc_adapter.connect_orbital_client", return_value=connection), patch(
            "mission_control.krpc_adapter.KRPCOrbitalAdapter", return_value=self.adapter
        ), patch("scripts.return_to_ground.parachute_plan", return_value={}), patch(
            "scripts.return_to_ground.run_parachute_return", return_value={"status": "succeeded"}
        ) as run:
            self.assertEqual(main(["--execute"]), 0)
        mission, adapter = run.call_args.args
        self.assertEqual(mission, ParachuteReturnMission.from_dict(mission.to_dict()))
        self.assertIs(adapter, self.adapter)
        connection.close.assert_called_once()
