# RunPod deployment notes

## Why DinD on Pod B

RunPod GPU pods are themselves containers. By default there is no Docker daemon
inside the pod, but prime-rl (and our `DockerEnvironment`) needs to spawn
per-task rollout containers. Two options exist:

1. **Docker-in-Docker** — run `dockerd` inside the pod. Requires `--privileged`,
   and the inner daemon's storage lives in the pod's overlay filesystem, which
   is slow and re-downloads images on every pod re-create.
2. **Host socket bind-mount** — run the pod with `-v /var/run/docker.sock:/var/run/docker.sock`
   so containers the pod spawns are actually siblings on the host daemon.
   Simpler, faster image reuse via the network volume cache, and crucially the
   containers live on the *host* so they survive the pod-container being kicked.

**We use option 2.** It's simpler and more reliable at the cost of coupling the
pod to the host's Docker version. RunPod's documented socket path is
`/var/run/docker.sock`.

## Pod-create flags

Pod A (CPU datagen):
```
runpodctl pod create \
  --image ml-systems/datagen:latest \
  --gpu-count 0 \
  --cpu 8 \
  --memory-gb 32 \
  --volume-path /workspace \
  --volume-size-gb 500 \
  --env NEBIUS_API_KEY=... --env GITHUB_TOKEN=... \
  --mount-sock     # RunPod extension that binds /var/run/docker.sock
```

Pod B (H100, privileged, DinD-via-sock):
```
runpodctl pod create \
  --image ml-systems/train:latest \
  --gpu H100_SXM --gpu-count 1 \
  --cpu 16 \
  --memory-gb 128 \
  --volume-path /workspace \
  --volume-size-gb 500 \
  --privileged \
  --mount-sock \
  --shm-size 16g
```

If `--mount-sock` is unavailable on the pod-create plan, use the Dockerfile's
volume mount at `docker compose` time instead (see
`docker-compose.train.yml`).

## Volume handoff

1. Pod A finishes → `/workspace/datasets/pilot/pilot.jsonl`,
   `/workspace/datasets/pilot/heldout.jsonl`, `/workspace/docker-cache/*.tar.gz`
   on the network volume.
2. Stop Pod A. Volume detaches.
3. Create Pod B and attach the same network volume at `/workspace`.
4. On Pod B: `docker load < /workspace/docker-cache/*.tar.gz` (~3 min).
5. Run `docker compose -f infra/docker-compose.train.yml up train`.
6. After training: `down`, then
   `docker compose -f infra/docker-compose.evaluate.yml up evaluate`.

## Kill-test procedure

At step 12 (ckpt_interval=10, so step 10 checkpoint already exists):
1. Record trajectory hashes for steps 11 & 12 from `sessions/train-*/logs/trace.jsonl`.
2. `docker kill ml-train` (the supervisor script inside the pod will restart it).
3. On restart the supervisor invokes `uv run train --resume latest` →
   resumes from step_10 checkpoint → re-advances to step 12.
4. Assert the post-resume hashes for steps 11 & 12 match the pre-kill values.

This catches RNG-state drift that a plain checkpoint-reload test would miss.

## Known gotchas

- **Host kernel mismatch** with the rollout image can break `git apply` — pin
  the rollout-image git version (already pinned in `rollout.Dockerfile`).
- **Docker socket permissions**: the pod user needs gid=999 (docker group on
  host) or the bind will fail `Permission denied`. Adjust Dockerfile RUN
  `usermod -a -G docker` if you see this.
- **GPU visibility**: use `nvidia-smi` inside the pod to confirm the H100 is
  visible before `docker compose up`. If it's not, the pod was created without
  the NVIDIA runtime.
