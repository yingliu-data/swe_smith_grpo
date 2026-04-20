from __future__ import annotations

import json
from pathlib import Path

from .swebench_jsonl import InstanceRecord


class HarborDirWriter:
    """Serialise an InstanceRecord into a Harbor-style directory.

    Layout:
      <root>/<instance_id>/
        instruction.md
        test.sh
        reference.diff
        task.json
    """

    def __init__(self, root: Path | str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def write(self, rec: InstanceRecord, test_command: list[str]) -> Path:
        dest = self._root / rec.instance_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "instruction.md").write_text(rec.problem_statement.strip() + "\n", encoding="utf-8")
        (dest / "reference.diff").write_text(rec.patch, encoding="utf-8")
        test_sh = "#!/usr/bin/env bash\nset -e\n" + " ".join(_shell_quote(a) for a in test_command) + "\n"
        script = dest / "test.sh"
        script.write_text(test_sh, encoding="utf-8")
        script.chmod(0o755)
        task = {
            "repository": rec.repo,
            "base_commit": rec.base_commit,
            "test_command": list(test_command),
            "metadata": rec.metadata or {},
        }
        (dest / "task.json").write_text(json.dumps(task, indent=2, sort_keys=True), encoding="utf-8")
        return dest


def _shell_quote(arg: str) -> str:
    if not arg or any(c in arg for c in ' \t"\\$`'):
        esc = arg.replace("'", "'\\''")
        return f"'{esc}'"
    return arg
