"""Does the grown student actor's own weights change during training?"""
import torch, pathlib
from projects.impedance_instron.bilateral.policy import Actor
from projects.impedance_instron.bilateral.adaptation import grow_actor, EMBEDDING_NAMES
from projects.impedance_instron.bilateral.shoe_randomization import THETA_OBSERVATION_NAMES
import json

out = pathlib.Path("outputs/bilateral/grf_spike_2026_10_14/tmp_ab_student")
teach = pathlib.Path("Z:/newton_grid/runs/shoe_adapt/16701746/02_teacher_asym/policy_teacher.pt")
saved = torch.load(teach, map_location="cpu", weights_only=True)
names = json.loads(teach.with_suffix(".json").read_text(encoding="utf-8"))["observations"]
import numpy as np
meta = json.loads(teach.with_suffix(".json").read_text(encoding="utf-8"))
source = Actor(tuple(saved["hidden"]), observation_names=tuple(names), action_names=tuple(meta["actions"]), seed_logits=np.zeros(len(meta["actions"]), dtype=np.float32))
source.load_state_dict(saved["state_dict"])
grown = grow_actor(source, EMBEDDING_NAMES, drop_names=THETA_OBSERVATION_NAMES)
init = {k: v.clone() for k, v in grown.state_dict().items() if k.startswith("net.")}
for tag in ("policy.pt",):
    f = out / tag
    if not f.exists():
        print("missing", tag); continue
    got = torch.load(f, map_location="cpu", weights_only=True)["state_dict"]
    d = max(float((got[k] - init[k]).abs().max()) for k in init)
    print(f"{tag}: max |weight - grown init| = {d:.3e}  (0. means the actor never moved)")
