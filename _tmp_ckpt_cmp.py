import torch, pathlib
d = pathlib.Path("Z:/newton_grid/runs/spatial_mujoco/16685657/00_ft100_mujoco")
print("files:", sorted(p.name for p in d.iterdir()))
ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=True)
print("checkpoint keys:", sorted(ck.keys()), "iteration:", ck.get("iteration"))
actor_ck = {k.removeprefix("actor."): v for k, v in ck["model"].items() if k.startswith("actor.")}
for tag in ("policy_01500.pt", "policy.pt"):
    f = d / tag
    if not f.exists():
        print(tag, "MISSING"); continue
    pol = torch.load(f, map_location="cpu", weights_only=True)
    sd = pol["state_dict"]
    same = set(sd) == set(actor_ck) and all(torch.equal(sd[k], actor_ck[k]) for k in sd)
    print(f"{tag}: same actor weights as checkpoint.pt ->", same, "| keys:", len(sd), "| has log_std:", "log_std" in pol)
