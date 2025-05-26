"""Retrieve issue ticker and add in commit message."""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_ticket_id_from_branch_name(branch: str) -> str | None:
    """Get first number from branch name."""
    matches = re.findall(r"[0-9]+", branch)
    return matches[0] if matches else None


def is_git_in_special_state() -> bool:
    """Check if git is in rebase, merge, or detached HEAD state."""
    git_dir = Path(".git")
    return (
        (git_dir / "rebase-apply").exists()
        or (git_dir / "rebase-merge").exists()
        or (git_dir / "MERGE_HEAD").exists()
    )


def get_branch_name() -> str | None:
    """Get current branch name, or None if in detached HEAD."""
    try:
        return subprocess.check_output(  # noqa: S603
            ["/usr/bin/git", "symbolic-ref", "--short", "HEAD"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    """CLI for retrieving issue ticker and adding it to the commit message."""
    parser = argparse.ArgumentParser()
    parser.add_argument("commit_msg_filepath")
    parser.add_argument(
        "-t",
        "--template",
        default="[{}]",
        help="Template to render ticket ID into",
    )
    args = parser.parse_args()

    commit_msg_filepath = Path(args.commit_msg_filepath)
    template = args.template

    try:
        with commit_msg_filepath.open("r", encoding="utf-8") as f:
            content = f.readlines()

            if not all(line.startswith("#") for line in content if line.strip() != ""):
                return 0
    except Exception:
        logger.exception("Failed to update commit message")
        return 1

    branch = get_branch_name()
    if branch is None:
        # Detached HEAD or unable to resolve branch; skip hook
        return 0

    issue_id = get_ticket_id_from_branch_name(branch)
    if issue_id:
        prefix = template.format(f"#{issue_id.upper()}")
    else:
        prefix = template.format(branch)

    try:
        with commit_msg_filepath.open("r+", encoding="utf-8") as f:
            content = f.read()
            subject = content.split("\n", 1)[0].strip()
            f.seek(0)
            if prefix not in subject:
                f.write(f"{prefix}: {content}")
            else:
                f.write(content)
    except Exception:
        logger.exception("Failed to update commit message")
        return 1

    return 0


if __name__ == "__main__":
    # Skip hook during rebase, merge, or detached HEAD
    if is_git_in_special_state():
        sys.exit(0)
    sys.exit(main())
