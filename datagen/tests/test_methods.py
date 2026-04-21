from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from datagen.methods import Context
from datagen.methods.pr_mirror import PRMirrorMethod
from datagen.methods.procedural import ProceduralMethod


_REF_PATCH_1HUNK = """\
diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,2 @@
-def f(x):
-    return x + 1
+def f(x):
+    return x - 1
"""

_REF_PATCH_2HUNK = """\
diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,2 @@
-def f(x):
-    return x + 1
+def f(x):
+    return x - 1
@@ -10,2 +10,2 @@
-def g(y):
-    return y * 2
+def g(y):
+    return y / 2
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    d = tmp_path / "r"
    d.mkdir()
    (d / "mod.py").write_text("def f(x):\n    return x + 1\n")
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=d, check=True)
    return d


@pytest.mark.asyncio
async def test_pr_mirror_single_hunk_returns_none(repo: Path):
    # A reference patch with only one src hunk cannot be subsetted (dropping it
    # leaves nothing). The drop-one-hunk method returns None in that case.
    ctx = Context(
        repo="ex/x", repo_dir=repo, pr_number=1, base_commit="HEAD",
        merge_commit="HEAD", pr_title="t", pr_body="", reference_patch=_REF_PATCH_1HUNK,
        seed=42, trial_index=0,
    )
    cand = await PRMirrorMethod().generate(ctx)
    assert cand is None


@pytest.mark.asyncio
async def test_pr_mirror_drops_one_hunk(repo: Path):
    # With two hunks, pr_mirror keeps exactly one — a valid subset of the forward
    # fix that applies at base and leaves the other bug in place.
    ctx = Context(
        repo="ex/x", repo_dir=repo, pr_number=1, base_commit="HEAD",
        merge_commit="HEAD", pr_title="t", pr_body="", reference_patch=_REF_PATCH_2HUNK,
        seed=42, trial_index=0,
    )
    cand = await PRMirrorMethod().generate(ctx)
    assert cand is not None
    # Verify the output is a forward-direction diff (keeps the `+` fix lines,
    # not inverted). Exactly one of the two hunk-specific markers must appear.
    has_a = "return x - 1" in cand.buggy_patch and "+    return x - 1" in cand.buggy_patch
    has_b = "return y / 2" in cand.buggy_patch and "+    return y / 2" in cand.buggy_patch
    assert has_a ^ has_b, "drop-one-hunk must keep exactly one hunk in forward direction"


_REF_PATCH_MULTISECTION = """\
diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -18,5 +18,5 @@
 ctx_a
 ctx_b
-old_early
+new_early
 ctx_c
 ctx_d
diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -78,3 +78,3 @@
 ctx_e
-old_late
+new_late
 ctx_f
"""


@pytest.mark.asyncio
async def test_pr_mirror_preserves_multisection_structure(repo: Path):
    # git-format-patch output for multi-commit PRs yields MULTIPLE `diff --git`
    # sections for the same file; each section's line numbers are relative to
    # the state produced by previous sections. Consolidating them under one
    # header scrambles hunk ordering and produces
    # `patch fragment without header at line N`. Verify pr_mirror keeps each
    # original `diff --git` section intact.
    ctx = Context(
        repo="ex/x", repo_dir=repo, pr_number=1, base_commit="HEAD",
        merge_commit="HEAD", pr_title="t", pr_body="",
        reference_patch=_REF_PATCH_MULTISECTION, seed=42, trial_index=0,
    )
    cand = await PRMirrorMethod().generate(ctx)
    assert cand is not None
    # Exactly one of the two sections should be dropped — by the LAST-section
    # safety rule, that's the single-hunk second section.
    assert cand.buggy_patch.count("diff --git a/mod.py") == 1
    assert "old_early" in cand.buggy_patch  # first section kept
    assert "old_late" not in cand.buggy_patch  # second section dropped


@pytest.mark.asyncio
async def test_procedural_mutation_produces_diff(repo: Path):
    ctx = Context(
        repo="ex/x", repo_dir=repo, pr_number=1, base_commit="HEAD",
        merge_commit="HEAD", pr_title="t", pr_body="", reference_patch=_REF_PATCH_1HUNK,
        seed=42, trial_index=0,
    )
    cand = await ProceduralMethod().generate(ctx)
    assert cand is not None
    assert cand.buggy_patch.startswith("diff --git a/mod.py")
    # New implementation uses text surgery, not ast.unparse — verify that the
    # mutated file is still parseable (no syntax regression) by checking the
    # diff replaces the BinOp operator on a single line.
    assert "-    return x + 1" in cand.buggy_patch
    assert "+    return x - 1" in cand.buggy_patch
