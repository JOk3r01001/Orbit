"""Plan a return, capture a descent save, or execute deorbit-to-PPO handoff."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

from mission_control.descent import ReturnMission, braking_metrics
from mission_control.supervisor import ControlCommand
from scripts.orbit_maneuver import plan_report, read_snapshot, run_live


class TelemetryClock:
    """Bound paused, rewound and jumping simulation time in the live runner."""

    def __init__(self, ut_s):
        self.ut_s = ut_s
        self.last_wall = time.monotonic()

    def advance(self, ut_s):
        if ut_s < self.ut_s or ut_s - self.ut_s > 2:
            raise RuntimeError("Return telemetry rewound or exceeded a 2-second gap")
        if ut_s == self.ut_s:
            if time.monotonic() - self.last_wall > 3:
                raise TimeoutError("Return simulation stopped advancing")
            return False
        self.ut_s = ut_s
        self.last_wall = time.monotonic()
        return True


def return_plan(mission, state):
    result = plan_report(mission.deorbit, state)
    result["return_mission"] = mission.to_dict()
    result["note"] += (" Periapsis is a trajectory target, not a handoff trigger or landing-site prediction."
                       " No atmospheric, heating, terrain, or propellant-feasibility prediction is made.")
    return result


def run_return(mission, adapter, descent, model=None, capture_save=None, log_dir=Path("training_output/returns")):
    """One sequential control owner; capture and inference never learn online."""
    if (model is None) == (capture_save is None):
        raise ValueError("Choose a trained model or a handoff-save capture")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"return_{time.time_ns()}.json"
    telemetry = path.with_suffix(".jsonl")
    report = {"status": "running", "phase": "deorbit", "mission": mission.to_dict()}
    try:
        # Record a single mission fuel baseline, and bind all telemetry to the
        # original craft. No save reload is allowed during this return.
        descent.vessel = adapter.vessel
        descent.body = adapter.body
        descent.initial_fuel = adapter.initial_fuel
        descent.initial_parts = len(adapter.vessel.parts.all)
        initial = descent.read()
        if not initial.intact or initial.acceleration_mps2 < mission.profile.min_thrust_weight * initial.gravity_mps2:
            raise ValueError("Return preflight requires an intact craft and sufficient active-engine thrust/weight")
        if any(chute.deployed for chute in adapter.vessel.parts.parachutes):
            raise ValueError("Stow/remove parachutes before a powered return")
        report["deorbit"] = run_live(mission.deorbit, adapter, log_dir=log_dir)
        if report["deorbit"]["result"]["status"] != "succeeded":
            report["status"] = "aborted" if report["deorbit"].get("interrupted") else "failed"
            report["reason"] = "Deorbit did not complete; no handoff attempted"
            return report
        report["phase"] = "coast"
        print("Phase: return coast | engines off; monitoring descent handoff", flush=True)
        sample = descent.read()
        coast_start = sample.ut_s
        clock = TelemetryClock(sample.ut_s)
        with telemetry.open("x", encoding="utf-8") as stream:
            while True:
                if sample.ut_s - coast_start > mission.coast_timeout_s:
                    raise TimeoutError("Return coast timed out")
                reserve = max(mission.deorbit.minimum_fuel_fraction,
                              mission.landing.minimum_fuel_fraction if mission.landing else 0)
                if sample.fuel_fraction < reserve:
                    raise RuntimeError("Return fuel reserve violated before handoff")
                state = mission.handoff(sample)
                stream.write(json.dumps({"phase": "coast", "sample": asdict(sample), "handoff": state}, allow_nan=False) + "\n")
                stream.flush()
                if state == "ready":
                    break
                # Stay surface-retrograde at zero throttle during passive entry.
                adapter.apply(ControlCommand(attitude="surface_retrograde"))
                while True:
                    time.sleep(.1)
                    sample = descent.read()
                    if clock.advance(sample.ut_s):
                        break
            adapter.stop()  # Release orbital ownership before PPO/capture owns it.
            # Recheck after the stop RPCs; do not capture/handoff stale telemetry.
            sample = descent.read()
            if mission.handoff(sample) != "ready":
                raise RuntimeError("Descent left the handoff window during transfer")
            report["handoff"] = {"sample": asdict(sample), "braking": braking_metrics(sample, mission.profile)}
            if capture_save is not None:
                descent.sc.save(capture_save)
                report.update(status="handoff_saved", phase="captured", save=capture_save)
                print(f"Handoff descent saved as {capture_save}; no landing policy was run.", flush=True)
                return report
            report["phase"] = "landing"
            observation, _ = descent.begin_current(expected_vessel=adapter.vessel, fuel_reference=adapter.initial_fuel)
            print("Phase: landing PPO | deorbit controller released", flush=True)
            total = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = descent.step(action)
                total += reward
                stream.write(json.dumps({"phase": "landing", "sample": asdict(descent.previous),
                                         "reward": reward, "info": info}, allow_nan=False) + "\n")
                stream.flush()
                if terminated or truncated:
                    report.update(status="succeeded" if info["is_success"] else "failed",
                                  reason=info["reason"], landing_reward=total, landing=info)
                    break
    except KeyboardInterrupt:
        report.update(status="aborted", reason="Interrupted by user")
    except Exception as exc:
        report.update(status="failed", reason=str(exc))
    finally:
        try:
            adapter.stop()
        except Exception as exc:
            report.update(status="failed", cutoff_error=str(exc))
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Return report saved: {path}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=Path, default=Path("examples/return/mission.json"))
    parser.add_argument("--snapshot", type=Path, help="Read-only offline orbital snapshot")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", type=Path, default=Path("training_output/descent_v2/descent_latest.zip"))
    parser.add_argument("--capture-handoff", help="Instead of running PPO, save the first valid handoff under this KSP name")
    args = parser.parse_args(argv)
    if args.snapshot and args.execute:
        parser.error("--snapshot cannot execute flight controls")
    if args.capture_handoff and (not args.execute or not re.fullmatch(r"[A-Za-z0-9_-]+", args.capture_handoff)):
        parser.error("Capture requires --execute and an alphanumeric/underscore/hyphen save name")
    try:
        mission = ReturnMission.from_dict(json.loads(args.mission.read_text(encoding="utf-8")))
        if args.snapshot:
            print(json.dumps(return_plan(mission, read_snapshot(args.snapshot)), indent=2, allow_nan=False))
            return 0
        # Load/validate weights BEFORE connecting or commanding the deorbit.
        model = None
        if args.execute:
            from scripts.descent_train import DescentEnv, load_policy
            if not args.capture_handoff:
                model = load_policy(args.model, mission.profile)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    from mission_control.krpc_adapter import KRPCOrbitalAdapter, connect_orbital_client
    connection = adapter = None
    try:
        connection = connect_orbital_client()
        adapter = KRPCOrbitalAdapter(connection)
        print(json.dumps(return_plan(mission, adapter.read()), indent=2, allow_nan=False), flush=True)
        if not args.execute:
            return 0
        descent = DescentEnv(task=mission.landing, profile=mission.profile, connection=connection)
        report = run_return(mission, adapter, descent, model, args.capture_handoff)
        print(json.dumps({key: report.get(key) for key in ("status", "phase", "reason", "save")}, indent=2))
        return 0 if report["status"] in {"succeeded", "handoff_saved"} else 1
    finally:
        try:
            if adapter is not None and args.execute:
                adapter.stop()
        finally:
            if connection is not None:
                connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
