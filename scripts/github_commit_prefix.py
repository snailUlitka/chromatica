"""Retrieve issue ticker and add in commit message."""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_ticket_id_from_branch_name(branch) -> str | None:
    """Get first number from branch name."""
    matches = re.findall("[0-9]+", branch)

    if len(matches) > 0:
        return matches[0]

    return None


def main() -> int:
    """CLI for retrieve issue ticker and add in commit message."""
    parser = argparse.ArgumentParser()
    parser.add_argument("commit_msg_filepath")
    parser.add_argument(
        "-t",
        "--template",
        default="[{}]",
        help="Template to render ticket id into",
    )
    args = parser.parse_args()
    commit_msg_filepath = Path(args.commit_msg_filepath)
    template = args.template

    branch = ""
    try:
        branch = subprocess.check_output(  # noqa: S603
            ["/usr/bin/git", "symbolic-ref", "--short", "HEAD"],
            universal_newlines=True,
        ).strip()
    except Exception:
        logger.exception("Error on ticker retrive")
        return 1

    result = get_ticket_id_from_branch_name(branch)
    issue_number = ""

    if result:
        issue_number = result.upper()
        prefix = template.format("#" + issue_number)
    else:
        prefix = template.format(branch)

    with commit_msg_filepath.open("r+") as f:
        content = f.read()
        content_subject = content.split("\n", maxsplit=1)[0].strip()
        f.seek(0, 0)
        if prefix not in content_subject:
            f.write(f"{prefix}: {content}")
        else:
            f.write(content)

    return 0


if __name__ == "__main__":
    sys.exit(main())
