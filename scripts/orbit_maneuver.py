"""Preview or execute a change_orbit mission on the current vessel in orbit."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from mission_control.commands import MissionTask, TaskType, number
from mission_control.orbital_controller import DeorbitController, OrbitalManeuverController, elements_from_state
from mission_control.orbits import OrbitalTelemetry, STANDARD_GRAVITY, burn_duration, plan_deorbit, plan_orbit_change
from mission_control.supervisor import FlightState, MissionSupervisor, Status


def read_snapshot(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "orbital" not in payload:
        raise ValueError("Snapshot must be a FlightState object with orbital telemetry")
    payload["orbital"] = OrbitalTelemetry(**payload["orbital"])
    return FlightState(**payload)


def plan_report(task, state):
    if task.task not in {TaskType.CHANGE_ORBIT, TaskType.DEORBIT}:
        raise ValueError("This runner accepts change_orbit or deorbit missions only")
    if state.body != task.body or state.situation != "orbiting":
        raise ValueError("The current vessel must be orbiting the requested body")
    if state.fuel_fraction < task.minimum_fuel_fraction:
        raise ValueError("The current fuel reserve is already below the request")
    elements = elements_from_state(state)
    burns = plan_deorbit(elements, task.target) if task.task == TaskType.DEORBIT else plan_orbit_change(elements, task.target)
    orbital = state.orbital
    mass = orbital.mass_kg
    result = []
    estimates_valid = not burns or (orbital.available_thrust_n > 0 and orbital.specific_impulse_s > 0)
    for burn in burns:
        duration = None
        if estimates_valid:
            duration = burn_duration(burn.delta_v_mps, mass, orbital.available_thrust_n, orbital.specific_impulse_s)
            mass *= math.exp(-abs(burn.delta_v_mps) / (orbital.specific_impulse_s * STANDARD_GRAVITY))
        result.append({**asdict(burn), "attitude": burn.attitude, "estimated_duration_s": duration})
    return {
        "mode": "plan_only", "mission": task.to_dict(), "vessel_id": state.vessel_id,
        "current_orbit_m": {"apoapsis": state.apoapsis_m, "periapsis": state.periapsis_m},
        "burns": result, "total_delta_v_mps": sum(abs(burn.delta_v_mps) for burn in burns),
        "engine_estimates_available": bool(estimates_valid),
        "note": "Impulsive two-body estimates. Live control replans from telemetry; no fuel-feasibility guarantee.",
    }


def run_live(task, adapter, interval_s=0.2, log_dir=Path("training_output/maneuvers")):
    """Run one mission; errors, Ctrl+C, and every terminal state request cutoff."""
    number(interval_s, "interval_s", 0.05, 1)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    telemetry_path = log_dir / f"maneuver_{stamp}.jsonl"
    report_path = log_dir / f"maneuver_{stamp}.json"
    controller = DeorbitController() if task.task == TaskType.DEORBIT else OrbitalManeuverController()
    supervisor = MissionSupervisor({task.task: controller}, adapter)
    initial = adapter.read()
    initial_plan = plan_report(task, initial)
    interrupted = False
    state = initial
    try:
        with telemetry_path.open("x", encoding="utf-8") as stream:
            supervisor.submit(task, initial)
            last_ut = initial.ut_s
            last_advance_wall = time.monotonic()
            last_phase = None
            while supervisor.status == Status.RUNNING:
                time.sleep(interval_s)
                state = adapter.read()
                if state.ut_s < last_ut:
                    raise RuntimeError("Simulation time rewound")
                if state.ut_s == last_ut:
                    if time.monotonic() - last_advance_wall > 3:
                        raise RuntimeError("Simulation time stopped advancing")
                    continue
                if state.ut_s - last_ut > 2:
                    raise RuntimeError("Telemetry gap exceeded 2 simulation seconds")
                last_ut = state.ut_s
                last_advance_wall = time.monotonic()
                report = supervisor.tick(state)
                details = controller.report()
                stream.write(json.dumps({"state": asdict(state), "mission": report, "controller": details}, allow_nan=False) + "\n")
                stream.flush()
                if details["phase"] != last_phase:
                    print(f"Phase: {details['phase']} | AP {state.apoapsis_m / 1000:.2f} km | PE {state.periapsis_m / 1000:.2f} km", flush=True)
                    last_phase = details["phase"]
    except KeyboardInterrupt:
        interrupted = True
        if supervisor.status == Status.RUNNING:
            supervisor.abort()
    except Exception as exc:
        if supervisor.status == Status.RUNNING:
            supervisor.fail(f"Live runner failed: {exc}")
        else:
            raise
    finally:
        if supervisor.status == Status.RUNNING:
            supervisor.abort()
        result = {
            "mission": task.to_dict(), "result": supervisor.report(),
            "controller": controller.report(), "initial_plan": initial_plan,
            "final_orbit_m": {"apoapsis": state.apoapsis_m, "periapsis": state.periapsis_m},
            "final_fuel_fraction": state.fuel_fraction,
            "interrupted": interrupted, "telemetry_file": telemetry_path.name,
        }
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved: {report_path}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=Path, default=Path("examples/change_orbit.json"))
    parser.add_argument("--snapshot", type=Path, help="Offline FlightState JSON; never connects to KSP")
    parser.add_argument("--execute", action="store_true", help="Take control and perform the maneuver (default is read-only planning)")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--rpc-timeout", type=float, default=5.0)
    parser.add_argument("--log-dir", type=Path, default=Path("training_output/maneuvers"))
    args = parser.parse_args(argv)
    if args.snapshot and args.execute:
        parser.error("--snapshot cannot be combined with --execute")
    try:
        number(args.interval, "interval", 0.05, 1)
        number(args.rpc_timeout, "rpc_timeout", 0.1, 30)
        task = MissionTask.from_dict(json.loads(args.mission.read_text(encoding="utf-8")))
        if args.snapshot:
            print(json.dumps(plan_report(task, read_snapshot(args.snapshot)), indent=2, allow_nan=False))
            return 0
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    # Import the transport only for live telemetry; offline planning is stdlib-only.
    from mission_control.krpc_adapter import KRPCOrbitalAdapter, connect_orbital_client

    connection = adapter = None
    try:
        connection = connect_orbital_client(args.rpc_timeout)
        adapter = KRPCOrbitalAdapter(connection)
        report = plan_report(task, adapter.read())
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        if not args.execute:
            return 0
        result = run_live(task, adapter, args.interval, args.log_dir)
        print(json.dumps(result["result"], indent=2), flush=True)
        if result["interrupted"]:
            return 130
        return 0 if result["result"]["status"] == "succeeded" else 1
    except KeyboardInterrupt:
        print("Maneuver interrupted; stopping vessel controls.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Maneuver failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if adapter is not None and args.execute:
            try:
                adapter.stop()
            except Exception as exc:
                print(f"Engine cutoff could not be confirmed: {exc}", file=sys.stderr)
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
