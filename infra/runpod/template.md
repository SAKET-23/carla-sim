# RunPod template `carla-sim-pod`

Field values for RunPod → Templates → New Template (`Plan/04-gpu/runpod-setup.md` section 3). The image is built by
`.github/workflows/image.yml`; the tag is the git short SHA it was built from, never `latest`.

| Field | Value |
|---|---|
| Name | `carla-sim-pod` |
| Type | Pod |
| Container image | `ghcr.io/<owner>/carla-sim-pod:<tag>` |
| Registry credentials | GHCR username + classic PAT with `read:packages` (only if the package is private) |
| Start command | empty (the image `ENTRYPOINT`, `/usr/local/bin/entrypoint.sh`, runs) |
| Container disk | 60 GB |
| Volume disk | 0 (the network volume replaces it) |
| Volume mount path | `/workspace` |
| HTTP ports | none |
| TCP ports | `22` |
| Env `NVIDIA_DRIVER_CAPABILITIES` | `all` |
| Env `NVIDIA_VISIBLE_DEVICES` | `all` |
| Env `PUBLIC_KEY` | contents of `~/.ssh/id_ed25519.pub` on the Mac (ECC-29: custom images do not get it injected) |
| Secret `ANTHROPIC_API_KEY` | `{{ RUNPOD_SECRET_anthropic_api_key }}` (create the secret first under Settings → Secrets) |
| Env `CARLA_QUALITY` | `Low` (ECC-26; `Epic` only for camera dataset generation) |

`NVIDIA_DRIVER_CAPABILITIES=all` is also baked into the image; whether the template value alone takes effect is
unverified. Do not expose 8765 (Foxglove) or 2000–2002 (CARLA) as public ports: use `ssh -L` tunnels.
Never paste an API key or token into this file.
