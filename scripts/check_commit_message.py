"""Require a commit subject recognized by the release parser."""

import sys
from pathlib import Path

from semantic_release.commit_parser.conventional import ConventionalCommitParser


def main() -> int:
    subject = Path(sys.argv[1]).read_text().partition("\n")[0].rstrip()
    parser = ConventionalCommitParser()
    parsed = parser.parse_message(subject)
    if parsed is not None and any(text.strip() for text in parsed.descriptions):
        return 0
    print(
        "Use <type>(optional scope): <description>, for example "
        "fix(mlflow): show recorded session titles.\n"
        f"Allowed types: {', '.join(parser.options.allowed_tags)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
