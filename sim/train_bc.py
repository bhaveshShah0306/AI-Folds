"""Behaviour cloning: train a small neural network to copy the taught demonstrations.

The network sees the same state the environment returns (joint angles, gripper
tip, whether it's holding the sheet, sheet corner positions) and predicts the
teacher's 4-number action. It runs fine on a CPU.

    python teach.py --task half --episodes 100 --noise 0.3 --out demos/half.npz
    python train_bc.py --task half --demos demos/half.npz --out policies/half_bc.pt
    python train_bc.py --task half --load policies/half_bc.pt --eval 20 --view
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fold_env import FoldEnv


def load_demos(path):
    data = np.load(path)
    episodes = sorted({k.split("/")[0] for k in data.files})
    obs = np.concatenate([data[f"{ep}/obs"] for ep in episodes])
    act = np.concatenate([data[f"{ep}/action"] for ep in episodes])
    return obs, act, len(episodes)


HIDDEN = 256  # train_rl.py builds SAC's actor with the same size so it can start from this


class Policy(nn.Module):
    """MLP from normalised observation to action in [-1, 1]."""

    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs):
        return self(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()


def train(obs, act, epochs, lr=1e-3, batch=256, seed=0):
    torch.manual_seed(seed)
    policy = Policy(obs.shape[1], act.shape[1])
    policy.obs_mean[:] = torch.as_tensor(obs.mean(axis=0))
    policy.obs_std[:] = torch.as_tensor(obs.std(axis=0) + 1e-6)
    X = torch.as_tensor(obs, dtype=torch.float32)
    Y = torch.as_tensor(act, dtype=torch.float32)

    # Hold out 10% of samples to spot overfitting.
    perm = torch.randperm(len(X))
    n_val = max(1, len(X) // 10)
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        policy.train()
        order = tr[torch.randperm(len(tr))]
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = ((policy(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            policy.eval()
            with torch.no_grad():
                train_loss = ((policy(X[tr]) - Y[tr]) ** 2).mean().item()
                val_loss = ((policy(X[val]) - Y[val]) ** 2).mean().item()
            print(f"epoch {epoch:4d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}")
    return policy


def evaluate(policy, task, episodes, seed=10_000, view=False):
    """Run the policy in fresh, randomised scenes it never saw in training."""
    env = FoldEnv(task=task)
    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        info = {"is_success": False}
        for _ in range(env.max_steps):
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            if viewer is not None:
                viewer.sync()
                import time
                time.sleep(0.05)
            if terminated or truncated:
                break
        successes += info["is_success"]
        gaps = np.round(info["fold_distances"] * 100, 1)
        print(f"eval episode {ep}: success={info['is_success']}, fold gaps (cm)={gaps}")
    if viewer is not None:
        viewer.close()
    print(f"success rate: {successes}/{episodes}")
    return successes / episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="diagonal")
    parser.add_argument("--demos", type=Path, help="demonstrations recorded by teach.py")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("policies/bc.pt"))
    parser.add_argument("--load", type=Path, help="skip training and load this policy")
    parser.add_argument("--eval", type=int, default=20, help="evaluation episodes")
    parser.add_argument("--view", action="store_true", help="watch evaluation in the MuJoCo viewer")
    args = parser.parse_args()

    if args.load:
        saved = torch.load(args.load)
        policy = Policy(saved["obs_dim"], saved["act_dim"])
        policy.load_state_dict(saved["state_dict"])
    else:
        if args.demos is None:
            parser.error("--demos is required unless --load is given")
        obs, act, n = load_demos(args.demos)
        print(f"training on {len(obs)} steps from {n} demos")
        policy = train(obs, act, args.epochs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": policy.state_dict(), "obs_dim": obs.shape[1],
                    "act_dim": act.shape[1], "task": args.task}, args.out)
        print(f"saved policy to {args.out}")

    policy.eval()
    if args.eval:
        evaluate(policy, args.task, args.eval, view=args.view)


if __name__ == "__main__":
    main()
