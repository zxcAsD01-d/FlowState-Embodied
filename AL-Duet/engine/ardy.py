# -*- coding: utf-8 -*-
"""ARDY Core8 continuous-motion engine for AL-Duet.

Wraps the official NVIDIA ARDY runtime (``app.ardy_runtime.ArdyGenerator``
from the AL-0A checkout, wired via the ``ALDUET_AL0A`` environment variable).
The quality recipe is the audited clean path:

* ARDY produces *pose only* -- no root/heading/hand constraints, no posture
  stabilizers (those were the corruption source, not the model);
* the world root is driven by an external speed controller (see actors.py);
* a drifted stand history is re-seeded from a model-generated stand snapshot
  (``reset_stand``), never hand-keyed;
* N agents share ONE model instance and step in ONE diffusion pass
  (``window_batch``) -- that is the parallel-inference path.

Diffusion steps: the Core8 checkpoint is trained with ``num_base_steps: 10``;
the sampler only supports 2..10 (larger values crash in ``space_timesteps``).
Default 10 = the official value (steps 2..10 are quality-equivalent; ablation
summary ships with the project docs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# External research code (not redistributed here): the AL-0A checkout that
# carries ``app/ardy_runtime.py`` plus the LLM2Vec text embeddings under
# ``runs/v7_clip/text_emb``.  Point ALDUET_AL0A at it, e.g.
#     set ALDUET_AL0A=D:\path\to\AL-0A
# The ARDY model implementation + Core8 checkpoint come from NVIDIA's ARDY
# release and must be importable as the ``ardy`` package (ARDY_ROOT).
AL0A = Path(os.environ.get("ALDUET_AL0A", ""))
if not AL0A.is_dir():
    raise RuntimeError(
        "ALDUET_AL0A is not set: point it at the AL-0A checkout that "
        "contains app/ardy_runtime.py and runs/v7_clip/text_emb")
for _p in (str(AL0A), str(AL0A / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.ardy_runtime import ArdyGenerator  # noqa: E402

# Real LLM2Vec text embeddings (same files the whole project has been using).
EMB_DIR = AL0A / "runs" / "v7_clip" / "text_emb"
INTENTS = {
    "stand": "A_person_is_standing_still.npy",
    "walk": "A_person_is_walking.npy",
    "walk_fwd": "A_person_walks_forward.npy",
    "wave": "A_person_waves_with_their_right_hand.npy",
}
STAND_HEIGHT = 0.947


def load_bank(device) -> dict:
    """intent -> (1,1,4096) torch tensor of the frozen LLM2Vec embedding."""
    import torch
    out = {}
    for name, fname in INTENTS.items():
        e = np.load(EMB_DIR / fname).astype(np.float32).reshape(1, 1, -1)
        out[name] = torch.as_tensor(e, device=device)
    return out


def set_text(gen, e) -> None:
    import torch
    if not torch.is_tensor(e):
        e = torch.as_tensor(np.asarray(e, np.float32), device=gen.model.device)
    if e.dim() == 1:
        e = e.reshape(1, 1, -1)
    gen._text_feat = e
    gen._text_mask = torch.ones(1, 1, dtype=torch.bool, device=gen.model.device)


def build_agents(n: int = 2, steps: int = 10, device: str = "cuda",
                 seed: int = 0) -> tuple[list, dict, list[str]]:
    """Load the model once and create N generators sharing it."""
    gens: list[ArdyGenerator] = []
    for i in range(n):
        g = ArdyGenerator(checkpoint="core8", device=device, steps=steps,
                          history_len=16, gen_len=8, cfg=(2.0, 2.0),
                          seed=seed + i)
        if i == 0:
            g.prepare()
        else:
            g.attach_model(gens[0].model)
        gens.append(g)
    bank = load_bank(gens[0].model.device)
    names = _skeleton_names(gens[0])
    return gens, bank, names


def _skeleton_names(gen) -> list[str]:
    skel = gen.model.skeleton
    ns = list(getattr(skel, "bone_order_names", []))
    if not ns:
        bi = skel.bone_index
        ns = sorted(bi, key=bi.get)
    return [str(n) for n in ns]


def seed_stand(gen, bank, frames: int = 8) -> dict:
    """One text-conditioned free window kept as the rolling-history prior.

    Returns the snapshot used by :func:`reset_stand` and a stand reference
    pose (the first frame's 27 local quaternions).
    """
    set_text(gen, bank["stand"])
    chunk = gen.warm_start_free(frames=frames)
    snap = {
        "h": gen.history_feats.clone(),
        "j": gen.joints.clone(),
        "hl": int(gen.history_len),
    }
    ref = SimpleNamespace(
        chunks=[chunk],
        rotations=np.asarray(chunk.rotations, np.float32),
        roots=np.asarray(chunk.roots, np.float32),
        joints=np.asarray(chunk.joints, np.float32),
        fps=float(chunk.fps),
    )
    return {"snap": snap, "stand_ref": ref}


def reset_stand(gen, snap: dict, joints_at_root, stand_height: float) -> None:
    """Re-seed history with the model-generated stand, shifted to the agent's
    current local root.  Cures the slow autoregressive crouch/sink."""
    import torch
    j = snap["snap"]["j"].clone()
    last = j[-1, gen.layout.root_idx]
    j[..., 0] += float(joints_at_root[0]) - float(last[0])
    j[..., 1] += float(stand_height) - float(last[1])
    j[..., 2] += float(joints_at_root[2]) - float(last[2])
    gen.joints = j
    gen.history_feats = snap["snap"]["h"].clone()
    gen.history_len = int(snap["snap"]["hl"])


def window_batch(gens: list, seed: int = 0) -> list:
    """One diffusion pass for ALL agents (parallel inference).

    Every generator must share one model, have equal history/gen lengths and
    already carry its text (the caller calls :func:`set_text` first).
    Returns one MotionChunk per agent, each agent's history advanced.
    """
    g0 = gens[0]
    for g in gens:
        if g.model is not g0.model:
            raise ValueError("batched generation needs a shared ARDY model")
        if g.history_feats is None:
            raise RuntimeError("call seed_stand() before window_batch()")
    n_frames = g0.history_len + g0.gen_len
    import torch
    history = torch.cat([g.history_feats for g in gens], dim=0)
    text_feat = torch.cat([g._text_feat for g in gens], dim=0)
    text_mask = torch.cat([g._text_mask for g in gens], dim=0)
    started = __import__("time").perf_counter()
    out = g0._sample_batch(n_frames, None, None, history, text_feat,
                           text_mask, seed)
    from app.ardy_runtime import _to_joints
    inverse = _to_joints(g0.model, out)
    chunks = []
    for i, g in enumerate(gens):
        chunks.append(g._pack(out, inverse, i, g.gen_len,
                              action="large_motion", started=started,
                              self_state=None, note="free_batch",
                              text_conditioned=True))
    return chunks


def window(gen, text, bank, seed: int = 0):
    """Single-agent free window (used for warm-up and non-batch fallbacks)."""
    set_text(gen, bank[text])
    started = __import__("time").perf_counter()
    return gen._finish(None, None, "large_motion", None, started, None)
