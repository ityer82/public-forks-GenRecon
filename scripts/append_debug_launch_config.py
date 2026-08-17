#!/usr/bin/env python3
"""Append a VS Code debugpy launch config entry (for the exact command a
pipeline stage just ran) to a log file, so it can be pasted straight into
a .vscode/launch.json "configurations" array to re-debug that stage on the
same data.
"""
import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to debug_config.log to append to")
    parser.add_argument("--name", required=True, help="launch.json config name")
    parser.add_argument("--program", required=True, help="Absolute path to the python script")
    parser.add_argument("--cwd", required=True, help="Working directory for the debug session")
    parser.add_argument(
        "--python",
        default="${workspaceFolder}/.venv/bin/python",
        help="Interpreter path (defaults to the target repo's venv)",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Everything after -- is passed as program args")
    parsed = parser.parse_args()

    args = parsed.args
    if args and args[0] == "--":
        args = args[1:]

    config = {
        "name": parsed.name,
        "type": "debugpy",
        "request": "launch",
        "program": parsed.program,
        "console": "integratedTerminal",
        "justMyCode": False,
        "cwd": parsed.cwd,
        "python": parsed.python,
        "args": args,
    }

    with open(parsed.log, "a") as f:
        f.write(json.dumps(config, indent=4) + ",\n")


if __name__ == "__main__":
    sys.exit(main())
