"""Preview or fly a stock Kerbin deorbit and parachute recovery mission."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import time

from mission_control.descent import ReturnMission, braking_metrics
from mission_control.recovery import ParachuteReturnMission
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


def parachute_plan(mission, state):
    result = plan_report(mission.deorbit, state)
    result["return_mission"] = mission.to_dict()
    result["landing_method"] = "stock_parachutes"
    result["note"] += " The burn calculation does not predict entry heating, parachute sizing, or the touchdown location."
    return result


def run_parachute_return(mission, adapter, log_dir=Path("training_output/returns")):
    """Deorbit on the existing vessel, then watch stock chutes through touchdown."""
    config = mission.recovery
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"return_{time.time_ns()}.json"
    telemetry = path.with_suffix(".jsonl")
    report = {"status": "running", "phase": "preflight", "mission": mission.to_dict(),
              "landing_method": "stock_parachutes", "deploy_commands": []}
    controls_claimed = False
    try:
        parts = set(adapter.vessel.parts.all)
        chutes = list(adapter.vessel.parts.parachutes)
        if not chutes or any(chute.state.name != "stowed" for chute in chutes):
            raise ValueError("Start with at least one stock parachute and all parachutes stowed")
        flight = adapter.vessel.flight(adapter.body.reference_frame)
        report["parachute_count"] = len(chutes)
        report["phase"] = "deorbit"
        controls_claimed = True
        report["deorbit"] = run_live(mission.deorbit, adapter, log_dir=log_dir)
        deorbit_result = report["deorbit"]["result"]
        if deorbit_result["status"] != "succeeded":
            report.update(status="aborted" if report["deorbit"].get("interrupted") else "failed",
                          reason=f"Deorbit ended: {deorbit_result['reason']}")
            return report

        report["phase"] = "entry"
        report["telemetry_file"] = telemetry.name
        print("Phase: atmospheric entry | throttle zero; waiting for safe parachute deployment", flush=True)
        sample = adapter.read_recovery(parts, chutes, flight)
        start_ut = sample.ut_s
        clock = TelemetryClock(start_ut)
        previous = None
        contact = None
        deployed_indices = set()
        with telemetry.open("x", encoding="utf-8") as stream:
            while True:
                report["last_sample"] = asdict(sample)
                stream.write(json.dumps({"phase": report["phase"], "sample": report["last_sample"]}) + "\n")
                stream.flush()
                if not sample.intact:
                    raise RuntimeError("Vessel lost parts during return")
                if sample.ut_s - start_ut >= config.timeout_s:
                    raise TimeoutError("Parachute recovery timed out")
                if sample.situation in {"landed", "splashed"}:
                    if contact is None:
                        contact = sample
                        report["phase"] = "touchdown"
                        report["touchdown"] = {
                            "sample": asdict(contact),
                            "last_airborne": asdict(previous) if previous else None,
                        }
                        adapter.stop()
                        if previous is None or "deployed" not in previous.chute_states:
                            raise RuntimeError("Contact before a fully open parachute was observed in flight")
                        if not previous.speeds_within_limits(config):
                            raise RuntimeError("Pre-contact speed exceeded the recovery limits")
                        if sample.situation == "splashed" and not config.allow_splashdown:
                            raise RuntimeError("Splashdown is disabled for this mission")
                    if sample.situation != contact.situation or not sample.speeds_within_limits(config):
                        raise RuntimeError("Vessel did not remain settled after contact")
                    if sample.ut_s - contact.ut_s >= config.settle_time_s:
                        outcome = "Splashdown" if contact.situation == "splashed" else "Landing"
                        report.update(status="succeeded", phase="complete", reason=f"{outcome} settled under parachute recovery")
                        break
                elif contact is not None:
                    raise RuntimeError("Vessel bounced after contact")
                else:
                    if sample.in_deploy_window(config):
                        for index, (chute, state) in enumerate(zip(chutes, sample.chute_states)):
                            if state != "stowed" or index in deployed_indices:
                                continue
                            if adapter.deploy_parachute_if_safe(chute):
                                deployed_indices.add(index)
                                report["deploy_commands"].append({"index": index, "ut_s": sample.ut_s, "height_m": sample.height_m})
                                if report["phase"] == "entry":
                                    adapter.stop()  # Let the craft hang naturally under the chutes.
                                    report["phase"] = "parachutes"
                                    print(f"Phase: parachutes | deployment commanded at {sample.height_m:.0f} m AGL", flush=True)
                    if report["phase"] == "entry":
                        if any(state in {"semi_deployed", "deployed"} for state in sample.chute_states):
                            adapter.stop()
                            report["phase"] = "parachutes"
                        elif config.entry_attitude == "surface_retrograde":
                            adapter.apply(ControlCommand(attitude="surface_retrograde"))
                    previous = sample
                while True:
                    time.sleep(.2)
                    sample = adapter.read_recovery(parts, chutes, flight)
                    if clock.advance(sample.ut_s):
                        break
    except KeyboardInterrupt:
        report.update(status="aborted", reason="Interrupted by user")
    except Exception as exc:
        report.update(status="failed", reason=str(exc))
    finally:
        if controls_claimed:
            try:
                adapter.stop()
            except Exception as exc:
                report.update(status="failed", cutoff_error=str(exc))
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Return report saved: {path}", flush=True)
    return report


def return_plan(mission, state):
    result = plan_report(mission.deorbit, state)
    orbital = state.orbital
    gravity = orbital.gravitational_parameter / (orbital.body_radius_m + state.altitude_m) ** 2
    surface_gravity = orbital.gravitational_parameter / orbital.body_radius_m ** 2
    acceleration = orbital.available_thrust_n / orbital.mass_kg
    result["engine_preflight"] = {
        "mass_kg": orbital.mass_kg,
        "available_thrust_n": orbital.available_thrust_n,
        "available_acceleration_mps2": acceleration,
        "current_gravity_mps2": gravity,
        "current_thrust_weight": acceleration / gravity,
        "surface_thrust_weight_estimate": acceleration / surface_gravity,
        "minimum_current_thrust_weight": mission.profile.min_thrust_weight,
        "current_thrust_weight_ok": acceleration / gravity >= mission.profile.min_thrust_weight,
        "note": "Available thrust is at full throttle with the current active-engine configuration. "
                "Throttle zero is allowed. Surface estimate uses current mass/thrust and ignores engine pressure changes.",
    }
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
        thrust_weight = initial.acceleration_mps2 / initial.gravity_mps2
        report["preflight"] = {
            "sample": asdict(initial), "thrust_weight": thrust_weight,
            "minimum_thrust_weight": mission.profile.min_thrust_weight,
        }
        print(f"Return preflight | TWR {thrust_weight:.3f} / required {mission.profile.min_thrust_weight:.3f} "
              f"| available acceleration {initial.acceleration_mps2:.3f} m/s^2 "
              f"| gravity {initial.gravity_mps2:.3f} m/s^2 | intact {initial.intact}", flush=True)
        if not initial.intact:
            raise ValueError("Return preflight failed: vessel parts were lost during telemetry sampling")
        if thrust_weight < mission.profile.min_thrust_weight:
            raise ValueError(
                f"Return preflight failed: available thrust/weight {thrust_weight:.3f} "
                f"is below the configured minimum {mission.profile.min_thrust_weight:.3f}. "
                "Check that the intended landing engine is active, fueled, and its thrust limiter is correct. "
                "If thrust is already fully available, use a lighter craft or more landing thrust; "
                "PPO training cannot add engine thrust. Throttle may remain at zero."
            )
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
    parser.add_argument("--mission", type=Path, help="Return mission JSON (defaults to examples/return/parachute.json)")
    parser.add_argument("--snapshot", type=Path, help="Read-only offline orbital snapshot")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--powered", action="store_true", help="Use the experimental powered descent path")
    parser.add_argument("--model", type=Path, help="Use an experimental powered descent checkpoint")
    parser.add_argument("--capture-handoff", help="Capture an experimental powered descent training save")
    args = parser.parse_args(argv)
    if args.snapshot and args.execute:
        parser.error("--snapshot cannot execute flight controls")
    if args.capture_handoff and (not args.execute or not re.fullmatch(r"[A-Za-z0-9_-]+", args.capture_handoff)):
        parser.error("Capture requires --execute and an alphanumeric/underscore/hyphen save name")
    if args.capture_handoff and args.model:
        parser.error("Choose either --capture-handoff or --model")
    powered = args.powered or args.model is not None or args.capture_handoff is not None
    mission_path = args.mission or Path("examples/return/mission.json" if powered else "examples/return/parachute.json")
    mission_type = ReturnMission if powered else ParachuteReturnMission
    preview = return_plan if powered else parachute_plan
    try:
        mission = mission_type.from_dict(json.loads(mission_path.read_text(encoding="utf-8")))
        if args.snapshot:
            print(json.dumps(preview(mission, read_snapshot(args.snapshot)), indent=2, allow_nan=False))
            return 0
        # Load/validate weights BEFORE connecting or commanding the deorbit.
        model = None
        if args.execute and powered:
            from scripts.descent_train import DescentEnv, load_policy
            if not args.capture_handoff:
                model = load_policy(args.model or Path("training_output/descent_v2/descent_latest.zip"), mission.profile)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    from mission_control.krpc_adapter import KRPCOrbitalAdapter, connect_orbital_client
    connection = adapter = None
    try:
        connection = connect_orbital_client()
        adapter = KRPCOrbitalAdapter(connection)
        plan = preview(mission, adapter.read())
        if not powered:
            plan["parachute_states"] = [chute.state.name for chute in adapter.vessel.parts.parachutes]
        print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
        if not args.execute:
            return 0
        if powered:
            descent = DescentEnv(task=mission.landing, profile=mission.profile, connection=connection)
            report = run_return(mission, adapter, descent, model, args.capture_handoff)
        else:
            report = run_parachute_return(mission, adapter)
        print(json.dumps({key: report.get(key) for key in ("status", "phase", "reason", "save")}, indent=2))
        if report["status"] == "aborted":
            return 130
        return 0 if report["status"] in {"succeeded", "handoff_saved"} else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Return failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
