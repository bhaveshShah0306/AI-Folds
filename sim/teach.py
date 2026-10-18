"""Teach the SO-101 a fold as a list of steps, then record demonstrations.

Each step moves the gripper to a named point on the sheet (see POINTS in
fold_env.py), opens/closes it, or lifts it. A simple controller turns the
steps into the same 4-number actions a learned policy outputs, so every run
can be saved as a demonstration for training.

    python teach.py --task diagonal --view            # watch it in the MuJoCo viewer
    python teach.py --task half --episodes 50 --out demos/half.npz
    python teach.py --task diagonal --episodes 50 --cameras --out demos/diag_cam.npz
    python teach.py --task half --episodes 100 --noise 0.3 --out demos/half.npz   # for training
"""

import argparse
from pathlib import Path

import numpy as np

from fold_env import MAX_STEP, POINTS, FoldEnv

HOVER = 0.06    # height to travel at (m)
PINCH = 0.008   # tip height when grabbing the sheet
DROP = 0.02     # tip height when letting go on top of the sheet

# Steps: (command, point name or None). Edit these, or add a new list, to teach
# a different fold. "drop_on" aims slightly past the target so the sheet lands on it.
STEPS = {
    "diagonal": [
        ("hover", "far_left"),
        ("down", "far_left"),
        ("close", None),
        ("up", None),
        ("hover", "near_right"),
        ("drop_on", "near_right"),
        ("open", None),
        ("up", None),
        ("home", None),
    ],
    # One arm can't hold a whole edge, so fold in half corner by corner.
    "half": [
        ("hover", "far_right"),
        ("down", "far_right"),
        ("close", None),
        ("up", None),
        ("hover", "near_right"),
        ("drop_on", "near_right"),
        ("open", None),
        ("up", None),
        ("hover", "far_left"),
        ("down", "far_left"),
        ("close", None),
        ("up", None),
        ("hover", "near_left"),
        ("drop_on", "near_left"),
        ("open", None),
        ("up", None),
        ("home", None),
    ],
}


class StepTeacher:
    """Turns a list of steps into env actions, one step at a time."""

    def __init__(self, env, steps, tol=0.004):
        self.env = env
        self.steps = list(steps)
        self.tol = tol
        self.i = 0
        self.grip = -1.0
        self.wait = 0
        self.settle = 0
        self.goal = None

    @property
    def done(self):
        return self.i >= len(self.steps)

    def _goal_for(self, cmd, point):
        sheet = self.env.sheet_positions()
        tip = self.env.target
        if cmd in ("hover", "down", "drop_on"):
            xy = sheet[POINTS[point], :2].copy()
            if cmd == "drop_on":
                # Overshoot a little away from the sheet centre so the held edge
                # ends up covering the target point rather than stopping short of it.
                away = xy - sheet.mean(axis=0)[:2]
                xy += 0.01 * away / (np.linalg.norm(away) + 1e-9)
            z = {"hover": HOVER, "down": PINCH, "drop_on": DROP}[cmd]
            return np.array([*xy, z])
        if cmd == "up":
            return np.array([tip[0], tip[1], HOVER])
        if cmd == "home":
            return np.array([0.20, 0.0, 0.10])
        return None

    def act(self):
        if self.done:
            return np.array([0.0, 0.0, 0.0, self.grip])
        cmd, point = self.steps[self.i]
        if cmd in ("close", "open"):
            self.grip = 1.0 if cmd == "close" else -1.0
            holding = self.env.grasped_vertex is not None
            self.wait += 1
            # Keep the command until the jaws have actually grabbed/let go, then
            # one more step to settle (or give up after 10 steps if nothing to grab).
            if holding == (cmd == "close") or self.wait >= 10:
                self.settle += 1
                if self.settle >= 2:
                    self.wait = self.settle = 0
                    self.i += 1
            return np.array([0.0, 0.0, 0.0, self.grip])

        if self.goal is None:
            self.goal = self._goal_for(cmd, point)
        delta = self.goal - self.env.target
        if cmd == "down" and np.linalg.norm(delta) < 0.006:
            # Start closing during the last few millimetres of the descent, like a
            # person would; it also gives the policy a clear "close now" signal.
            self.grip = 1.0
        if np.linalg.norm(delta) < self.tol:
            self.goal = None
            self.i += 1
        move = np.clip(delta / MAX_STEP, -1.0, 1.0)
        return np.array([*move, self.grip])


def run_episode(env, steps, seed, viewer=None, cameras=None, noise=0.0):
    """Run the teacher once. With noise > 0 the executed moves are randomly nudged,
    but the recorded action is still the teacher's clean correction, so a policy
    trained on the data learns how to recover from small mistakes."""
    obs, _ = env.reset(seed=seed)
    teacher = StepTeacher(env, steps)
    rng = np.random.default_rng(seed)
    record = {"obs": [], "action": [], "reward": [], "next_obs": [], "terminated": [], "landmarks": []}
    images = {name: [] for name in cameras or []}
    renderer = None
    if cameras:
        import mujoco
        renderer = mujoco.Renderer(env.model, 240, 320)
    info = {"is_success": False}
    for _ in range(env.max_steps):
        action = teacher.act()
        if renderer is not None:
            for name in cameras:
                renderer.update_scene(env.data, camera=name)
                images[name].append(renderer.render())
        record["obs"].append(obs)
        record["action"].append(action)
        executed = action.copy()
        executed[:3] += rng.normal(0.0, noise, 3)
        obs, reward, terminated, truncated, info = env.step(executed)
        record["reward"].append(reward)
        record["next_obs"].append(obs)
        record["terminated"].append(terminated)
        record["landmarks"].append(info["landmarks"])
        if viewer is not None:
            viewer.sync()
            import time
            time.sleep(0.05)
        if terminated or truncated or (teacher.done and info["is_success"]):
            break
    if renderer is not None:
        renderer.close()
    out = {k: np.array(v, dtype=np.float32) for k, v in record.items()}
    out.update({f"image_{k}": np.array(v, dtype=np.uint8) for k, v in images.items()})
    return out, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="diagonal", choices=sorted(STEPS))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--cameras", action="store_true", help="also record laptop + wrist camera images")
    parser.add_argument("--out", type=Path, help="save demonstrations to this .npz file")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="randomly nudge executed moves (e.g. 0.3) so demos include recoveries")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = FoldEnv(task=args.task)
    cameras = ["laptop", "wrist"] if args.cameras else None
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    episodes, successes = [], 0
    for ep in range(args.episodes):
        demo, info = run_episode(env, STEPS[args.task], args.seed + ep, viewer, cameras, args.noise)
        successes += info["is_success"]
        dists = np.round(info.get("fold_distances", []) * 100, 1)
        print(f"episode {ep}: {len(demo['action'])} steps, success={info['is_success']}, "
              f"fold gaps (cm)={dists}")
        if info["is_success"]:
            episodes.append(demo)
    print(f"{successes}/{args.episodes} successful folds")

    if args.out and episodes:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for ep, demo in enumerate(episodes):
            for k, v in demo.items():
                flat[f"ep{ep}/{k}"] = v
        np.savez_compressed(args.out, **flat)
        print(f"saved {len(episodes)} successful demos to {args.out}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
