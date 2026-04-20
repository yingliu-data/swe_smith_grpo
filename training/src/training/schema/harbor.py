from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(slots=True)
class HarborPrompt:
    instance_id: str
    repo: str
    base_commit: str
    instruction: str
    test_command: list[str]
    reference_patch: str
    test_patch: str
    fail_to_pass: list[str]
    metadata: dict[str, Any]

    def to_prompt_dict(self) -> dict[str, Any]:
        """Shape expected by prime-rl's dataset loader."""
        return {
            "instance_id": self.instance_id,
            "prompt": self.instruction,
            "metadata": {
                "repository": self.repo,
                "base_commit": self.base_commit,
                "test_command": self.test_command,
                "reference_patch": self.reference_patch,
                "test_patch": self.test_patch,
                "FAIL_TO_PASS": self.fail_to_pass,
                **self.metadata,
            },
        }


class JsonlToHarborConverter:
    """Convert a SWE-bench JSONL file into a stream of HarborPrompts for prime-rl."""

    @staticmethod
    def iter_prompts(jsonl_path: Path | str) -> Iterator[HarborPrompt]:
        p = Path(jsonl_path)
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                test_cmd = ["python", "-m", "pytest", "-x", "--tb=short", *d["FAIL_TO_PASS"]]
                yield HarborPrompt(
                    instance_id=d["instance_id"],
                    repo=d["repo"],
                    base_commit=d["base_commit"],
                    instruction=d["problem_statement"],
                    test_command=test_cmd,
                    reference_patch=d.get("patch", ""),
                    test_patch=d.get("test_patch", ""),
                    fail_to_pass=d.get("FAIL_TO_PASS", []),
                    metadata=d.get("metadata", {}),
                )
