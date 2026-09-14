# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dispatch repo-local project commands without importing unselected tools."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from importlib import import_module


def dispatch(
    package: str,
    description: str,
    commands: Mapping[str, tuple[str, str]],
    argv: list[str] | None = None,
) -> None:
    """Forward a selected task to its existing module's argument parser.

    Args:
        package: Fully qualified project package name.
        description: Package-level help text.
        commands: Command names mapped to module names and help text.
        argv: Command and unchanged tool arguments; default to process arguments.
    """
    parser = argparse.ArgumentParser(prog=package, description=description)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command, (_, help_text) in commands.items():
        subcommands.add_parser(command, help=help_text, add_help=False)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return
    # Parse only the task name; the existing tool owns all remaining flags.
    selected = parser.parse_args(arguments[:1]).command
    module_name = commands[selected][0]
    original_argv = sys.argv
    try:
        sys.argv = [f"{package} {selected}", *arguments[1:]]
        import_module(module_name).main()
    finally:
        sys.argv = original_argv
