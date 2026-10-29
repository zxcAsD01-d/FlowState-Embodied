# FlowState-Embodied

Visually-driven walking for a frozen motion diffusion model (ARDY Core8):
a first-person RGB view is encoded by a frozen DINOv2 into metric depth and a
collision mask, fused into an online occupancy map, planned over with A*, and
executed as root-boundary conditions by the frozen ARDY generator. The result
drives a VRM avatar in a three.js scene with cannon-es physics.

```
eye RGB ──► DINOv2 (frozen) + depth head + collision head (trained heads)
              │
              ▼
      online occupancy map (accumulated, world frame)
              │  A* + Chaikin smoothing ──► continuous route
              ▼
      RouteTracker ──► 8×3 body-frame root path per window
              │            (heading shaping: ≤0.35 rad/window closure,
              │             ±0.7 rad in-window clip, monotonic forward)
              ▼
      ARDY Core8 (frozen, 6-step DDIM, generate_path, stride-4 keyframes)
              │
              ▼
      VRM retarget (three-vrm) + cannon-es physics (draggable boxes)
```

## What is trained vs frozen

- **Frozen**: ARDY Core8 (`ARDY-Core-RP-20FPS-Horizon8`), DINOv2 ViT-B/14 reg4.
- **Trained here** (checkpoints included): the DINOv2 readout heads —
  absolute depth (`depth.pt`), collision mask (`boundary.pt`), and the fused
  online navigator (`nav.pt`, held-out absrel 0.171 / mask IoU 0.866).
- No ARDY LoRA / fine-tune; the motion model is used as-is.

## Findings that matter for motion quality

- **Keyframe stride**: conditioning the root/heading/height on *every* frame
  of the 8-frame window overspecifies the generation and progressively folds
  the upper body (head drops below the pelvis within ~12 windows). Sparse
  `constraint_stride=4` keyframes track the same path (0.032 m error) with no
  posture collapse.
- **Command shaping**: replanning storms used to command ±0.7 rad heading
  jumps every window with lateral drift — replaying the logged commands
  reproduces the crouch. The tracker now closes the yaw gap at ≤0.35
  rad/window, keeps per-frame heading offsets inside ±0.7 rad (the
  generator's training range), and integrates positions from the shaped
  headings so position/heading constraints stay self-consistent.
- Overshoot quality gates live in `tools/qc_*.py` (pose probe, nav
  closed-loop, block-and-replan).

## Run

```bash
pip install -r requirements.txt
python -m uvicorn ardy_v.service:app --host 127.0.0.1 --port 8830
# open http://127.0.0.1:8830/
```

- Click the floor to set a target; the avatar plans and walks there, stands
  on arrival.
- Boxes are physically simulated and mouse-draggable; the navigator fuses
  them into its occupancy map and replans when the route is blocked or a
  better route appears.
- HUD shows the first-person view, predicted depth, the online occupancy
  map, and per-window navigation state.

## External dependencies (not included)

| What | Where it comes from | Env var |
|---|---|---|
| ARDY Core8 generator (`run_r07_ardy_precision`, `run_r10_ardy_inverse`) | author's AL research tree — provide your own motion-generator stack exposing the same API | `FLOWSTATE_AL0A`, `FLOWSTATE_CIP_RUNS` |
| VRM avatar | any VRM 0/1 humanoid; place at `assets/avatar.vrm` | `FLOWSTATE_VRM` |
| DINOv2 ViT-B/14 reg4 weights | auto-downloaded by `torch.hub` cache on first run | — |
| YOLO weights (optional, only `tools/train_yolo_depth.py`) | download yourself (AGPL license) | `FLOWSTATE_YOLO_WEIGHTS` |

The demo avatar, dataset archives and the ARDY research tree are **not**
redistributed in this repository.

## Repo layout

```
ardy_v/       service (FastAPI), runtime, route tracker, occupancy, policy,
              depth/nav/yolo inference, data + training IO
web/          three.js + three-vrm + cannon-es front end (vendored)
tools/        data generation, training, QC probes (Playwright closed-loop)
checkpoints/  trained readout heads (depth / boundary / nav / yolo-depth)
docs/         demo screenshots
```

## Status and limits

- Closed-loop arrival and blocked-route replanning verified live
  (`tools/qc_nav_live.py`); the navigator's map can still hallucinate
  obstacles under some approach yaws (depth domain shift at eye height),
  which occasionally triggers extra replans — the planner recovers, but the
  map head would benefit from retraining on eye-height views.
- Core27 has no finger chains; manipulation is out of scope.
- The ARDY stack itself is external; without it this repo is the vision +
  planning + retarget half of the system.

## License

MIT — see [LICENSE](LICENSE). The vendored `web/vendor` libraries (three.js,
three-vrm, cannon-es) keep their own licenses; `checkpoints/*.pt` are
trained heads and provided as-is.
