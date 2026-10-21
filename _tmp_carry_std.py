"""Regenerate the teacher policy file so it carries the exploration std it trained with."""
import json, pathlib
import numpy as np
import torch
from projects.impedance_instron.bilateral.policy import Actor, save_policy, load_policy

src = pathlib.Path("Z:/newton_grid/runs/shoe_adapt/16701746/02_teacher_asym")
ck = torch.load(src / "checkpoint.pt", map_location="cpu", weights_only=True)
meta = json.loads((src / "policy.json").read_text(encoding="utf-8"))
full = ck["model"]
actor_sd = {k.removeprefix("actor."): v for k, v in full.items() if k.startswith("actor.")}
ref = torch.load(src / "policy.pt", map_location="cpu", weights_only=True)
print("keys in policy.pt:", sorted(ref["state_dict"])[:3], "n=", len(ref["state_dict"]))
print("keys grown actor :", sorted(actor_sd)[:3], "n=", len(actor_sd))
print("hidden:", meta["hidden"], "n_obs:", len(meta["observations"]), "n_act:", len(meta["actions"]))
print("weights identical to policy.pt:", all(torch.equal(ref["state_dict"][k], actor_sd[k]) for k in ref["state_dict"]))
print("hidden list identical:", ref["hidden"] == meta["hidden"])
actor = Actor(
    tuple(meta["hidden"]),
    observation_names=tuple(meta["observations"]),
    action_names=tuple(meta["actions"]),
    seed_logits=np.zeros(len(meta["actions"]), dtype=np.float32),
)
actor.load_state_dict(actor_sd)
out = src / "policy_teacher.pt"
save_policy(actor, out, {"note": f"run 16701746/02_teacher_asym checkpoint.pt at iteration {ck['iteration']}, with log_std"}, full["log_std"])
back = torch.load(out, map_location="cpu", weights_only=True)
print("written:", out.name, "| std mean:", float(full["log_std"].exp().mean()), "min:", float(full["log_std"].exp().min()), "max:", float(full["log_std"].exp().max()))
print("roundtrip has log_std:", "log_std" in back, "| log_std equal:", torch.equal(back["log_std"], full["log_std"].cpu()))
