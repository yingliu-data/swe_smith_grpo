from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

GITHUB_API = "https://api.github.com"


@dataclass(slots=True)
class PullRequestInfo:
    number: int
    base_commit: str
    merge_commit: str
    title: str
    body: str
    labels: list[str]
    patch_url: str


class RepoManager:
    def __init__(self, *, repos_root: Path, github_token: str | None = None):
        self._repos_root = repos_root
        self._token = github_token or os.environ.get("GITHUB_TOKEN")

    async def ensure_clone(self, repo: str) -> Path:
        slug = repo.replace("/", "__")
        target = self._repos_root / slug
        if (target / ".git").exists():
            return target
        self._repos_root.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(target),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"git clone {repo} failed: {err.decode()}")
        return target

    async def checkout(self, repo_dir: Path, commit: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_dir), "checkout", "-f", commit,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"git checkout {commit} failed: {err.decode()}")

    async def list_bug_prs(
        self, repo: str, *, labels: tuple[str, ...] = ("bug", "fix"), limit: int = 100
    ) -> list[PullRequestInfo]:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        owner, name = repo.split("/", 1)
        query = " ".join([
            f"repo:{repo}",
            "is:pr",
            "is:merged",
            f"label:{','.join(labels)}",
        ])
        url = f"{GITHUB_API}/search/issues"
        out: list[PullRequestInfo] = []
        page = 1
        auth_mode = "authenticated" if self._token else "ANONYMOUS (10 req/min)"
        print(f"[datagen] list_bug_prs: repo={repo} labels={labels} auth={auth_mode}", file=sys.stderr, flush=True)
        async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
            while len(out) < limit:
                resp = await client.get(url, params={"q": query, "per_page": 50, "page": page})
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                print(
                    f"[datagen] search page={page} status={resp.status_code} "
                    f"total_count={body.get('total_count')} items={len(body.get('items', []))} "
                    f"message={body.get('message')!r} "
                    f"rate_remaining={resp.headers.get('x-ratelimit-remaining')} "
                    f"rate_reset={resp.headers.get('x-ratelimit-reset')}",
                    file=sys.stderr, flush=True,
                )
                resp.raise_for_status()
                items = body.get("items", [])
                if not items:
                    break
                for item in items:
                    pr = await self._fetch_pr(client, owner, name, item["number"])
                    if pr is not None:
                        out.append(pr)
                        if len(out) >= limit:
                            break
                page += 1
        print(f"[datagen] list_bug_prs complete: {len(out)} PRs enumerated", file=sys.stderr, flush=True)
        return out

    async def _fetch_pr(
        self, client: httpx.AsyncClient, owner: str, name: str, number: int
    ) -> PullRequestInfo | None:
        resp = await client.get(f"{GITHUB_API}/repos/{owner}/{name}/pulls/{number}")
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data.get("merged_at") or not data.get("base"):
            return None
        return PullRequestInfo(
            number=number,
            base_commit=data["base"]["sha"],
            merge_commit=data["merge_commit_sha"],
            title=data.get("title", ""),
            body=data.get("body") or "",
            labels=[l["name"] for l in data.get("labels", [])],
            patch_url=data["patch_url"],
        )

    async def fetch_patch(self, pr: PullRequestInfo) -> str:
        headers = {"Accept": "application/vnd.github.v3.patch"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        async with httpx.AsyncClient(timeout=60.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(pr.patch_url)
            resp.raise_for_status()
            return resp.text
