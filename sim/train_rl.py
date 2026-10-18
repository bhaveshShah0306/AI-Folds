"""Reinforcement learning (SAC) that starts from what the arm already learned by imitation.

1. Load the behaviour-cloning policy from train_bc.py into SAC's actor, so SAC
   starts out folding as well as the imitation policy.
2. Copy the demonstrations into SAC's replay buffer.
3. Warm up: train only SAC's critic (which scores actions) on the demos, so an
   untrained critic doesn't wreck the good starting policy.
4. Practise in the simulator, keeping changes that increase the fold reward, while
   a small imitation term keeps the policy close to the demos (without it, early
   noisy RL updates quickly erase what imitation taught).

    python train_bc.py --task half --demos demos/half.npz --out policies/half_bc.pt
    python train_rl.py --task half --bc policies/half_bc.pt --demos demos/half.npz --out policies/half_sac
    python train_rl.py --task half --load policies/half_sac/best_model.zip --eval 20 --view

Runs on a CPU, but RL is slow: expect very roughly an hour per 100k steps.
The success rate is printed every --eval-every steps; best_model.zip keeps the best one.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from fold_env import FoldEnv
from train_bc import HIDDEN


def load_transitions(path):
    data = np.load(path)
    episodes = sorted({k.split("/")[0] for k in data.files})
    keys = ["obs", "action", "reward", "next_obs", "terminated"]
    return {k: np.concatenate([data[f"{ep}/{k}"] for ep in episodes]) for k in keys}, len(episodes)


def add_demos_to_buffer(model, demos):
    buf = model.replay_buffer
    for i in range(len(demos["obs"])):
        buf.add(
            demos["obs"][i][None], demos["next_obs"][i][None], demos["action"][i][None],
            np.array([demos["reward"][i]]), np.array([demos["terminated"][i]]), [{}],
        )


@torch.no_grad()
def load_bc_into_actor(model, bc_path, init_log_std=-2.0):
    """Copy a train_bc.py policy into SAC's actor.

    Both are obs -> 256 -> 256 -> tanh(action) networks. The BC policy normalises
    its input first; that is folded into the first layer's weights, so SAC's
    deterministic action is exactly the BC action.
    """
    saved = torch.load(bc_path)
    sd = saved["state_dict"]
    mean, std = sd["obs_mean"], sd["obs_std"]
    w0, b0 = sd["net.0.weight"] / std, sd["net.0.bias"] - (sd["net.0.weight"] / std) @ mean
    actor = model.actor
    actor.latent_pi[0].weight.copy_(w0)
    actor.latent_pi[0].bias.copy_(b0)
    actor.latent_pi[2].weight.copy_(sd["net.2.weight"])
    actor.latent_pi[2].bias.copy_(sd["net.2.bias"])
    actor.mu.weight.copy_(sd["net.4.weight"])
    actor.mu.bias.copy_(sd["net.4.bias"])
    # Small exploration noise to begin with, so practice stays close to what works.
    actor.log_std.weight.zero_()
    actor.log_std.bias.fill_(init_log_std)


def warm_up_critic(model, steps):
    """Train only the critic: the actor's weights are frozen while SAC trains."""
    model.set_logger(configure(folder=None, format_strings=[]))  # train() needs a logger
    params = list(model.actor.parameters())
    for p in params:
        p.requires_grad_(False)
    model.train(gradient_steps=steps, batch_size=model.batch_size)
    for p in params:
        p.requires_grad_(True)


class DemoRegulariser(BaseCallback):
    """After every SAC update, take one small step pulling the actor's deterministic
    action back towards the demo actions."""

    def __init__(self, demos, weight=1.0, batch=256):
        super().__init__()
        self.obs = torch.as_tensor(demos["obs"], dtype=torch.float32)
        self.act = torch.as_tensor(demos["action"], dtype=torch.float32)
        self.weight = weight
        self.batch = batch

    def _on_step(self):
        actor = self.model.actor
        idx = torch.randint(0, len(self.obs), (self.batch,))
        mean, _, _ = actor.get_action_dist_params(self.obs[idx].to(self.model.device))
        loss = self.weight * ((torch.tanh(mean) - self.act[idx].to(self.model.device)) ** 2).mean()
        actor.optimizer.zero_grad()
        loss.backward()
        actor.optimizer.step()
        return True


def evaluate(model, task, episodes, seed=20_000, view=False):
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
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if viewer is not None:
                viewer.sync()
                import time
                time.sleep(0.05)
            if terminated or truncated:
                break
        successes += info["is_success"]
    if viewer is not None:
        viewer.close()
    print(f"success rate: {successes}/{episodes}")
    return successes / episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="diagonal")
    parser.add_argument("--bc", type=Path, help="start from this train_bc.py policy")
    parser.add_argument("--demos", type=Path, help="demonstrations recorded by teach.py")
    parser.add_argument("--steps", type=int, default=200_000, help="simulator steps of practice")
    parser.add_argument("--warmup", type=int, default=5000, help="critic-only training steps on the demos")
    parser.add_argument("--demo-weight", type=float, default=10.0,
                        help="how strongly practice is pulled back towards the demos (0 = pure RL)")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("policies/sac"))
    parser.add_argument("--load", type=Path, help="skip training and load this model")
    parser.add_argument("--eval", type=int, default=20)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.load:
        model = SAC.load(args.load)
    else:
        env = Monitor(FoldEnv(task=args.task))
        model = SAC(
            "MlpPolicy", env, seed=args.seed, verbose=0,
            buffer_size=500_000, batch_size=256, learning_starts=0,
            learning_rate=1e-4, policy_kwargs={"net_arch": [HIDDEN, HIDDEN]},
            ent_coef="auto_0.01",
        )
        if args.bc:
            load_bc_into_actor(model, args.bc)
            print("starting from the imitation policy:", end=" ")
            evaluate(model, args.task, 10)
        if args.demos:
            demos, n = load_transitions(args.demos)
            print(f"loaded {len(demos['obs'])} steps from {n} demos into the replay buffer")
            add_demos_to_buffer(model, demos)
            if args.warmup:
                print(f"warming up the critic for {args.warmup} steps...")
                warm_up_critic(model, args.warmup)

        args.out.mkdir(parents=True, exist_ok=True)
        eval_cb = EvalCallback(
            Monitor(FoldEnv(task=args.task)), n_eval_episodes=10, eval_freq=args.eval_every,
            best_model_save_path=str(args.out), log_path=str(args.out), deterministic=True, verbose=1,
        )
        callbacks = [eval_cb]
        if args.demos and args.demo_weight > 0:
            callbacks.append(DemoRegulariser(demos, weight=args.demo_weight))
        model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks), progress_bar=False)
        model.save(args.out / "final_model")
        print(f"saved to {args.out}/ (best_model.zip is the best evaluation so far)")

    if args.eval:
        evaluate(model, args.task, args.eval, view=args.view)


if __name__ == "__main__":
    main()
