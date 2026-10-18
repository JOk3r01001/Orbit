"""Offline mission validation. This CLI does not connect to or control KSP."""

import argparse
import json
from pathlib import Path

from .commands import MissionTask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate"])
    parser.add_argument("mission", type=Path)
    parser.add_argument("--launch-options", action="store_true", help="Print options for the launch environment")
    args = parser.parse_args()
    try:
        task = MissionTask.from_dict(json.loads(args.mission.read_text(encoding="utf-8")))
        result = task.launch_options() if args.launch_options else {
            "status": "validated",
            "note": "Schema and profile limits checked; flight feasibility and controller availability are not checked.",
            "mission": task.to_dict(),
        }
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Mission rejected: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
