# AI-Folds

Teaching an SO-101 robot arm to fold paper and cloth, starting in simulation (MuJoCo).

## Setup

```bash
pip install -r requirements.txt
cd sim
```

## What's here

| Path | Purpose |
|---|---|
| `sim/models/so101/` | Official SO-101 MuJoCo model + STL meshes, with a wrist camera added |
| `sim/models/so101/fold_scene.xml` | Arm + 12x12 cm cloth sheet + `laptop`, `wrist`, `overhead` cameras |
| `sim/so101.py` | Arm landmarks (3D position of every joint + gripper tip) and inverse kinematics |
| `sim/fold_env.py` | Gymnasium environment for learning folds (RL or imitation) |
| `sim/teach.py` | Teach a fold as a list of steps, watch it, record demonstrations |
| `sim/train_bc.py` | Imitation learning: a neural network learns to copy the demonstrations |
| `sim/train_rl.py` | Reinforcement learning (SAC) starting from the imitation policy (experimental) |

## Try it

```bash
# Look at the arm, its meshes and the scene interactively
python -m mujoco.viewer --mjcf models/so101/fold_scene.xml

# Watch the arm do a taught fold
python teach.py --task diagonal --view
python teach.py --task half --view

# Record 50 successful demonstrations (add --cameras for laptop + wrist images)
python teach.py --task half --episodes 50 --out demos/half.npz
```

On macOS use `mjpython` instead of `python` for `--view`.

## Training (no GPU needed)

```bash
# 1. Record demonstrations; --noise nudges the moves so demos include recoveries
python teach.py --task half --episodes 200 --noise 0.15 --out demos/half.npz

# 2. Imitation learning: trains in about a minute on a CPU, then tests on 20 new sheets
python train_bc.py --task half --demos demos/half.npz --out policies/half_bc.pt --epochs 150

# Watch the trained policy fold
python train_bc.py --task half --load policies/half_bc.pt --eval 5 --view

# 3. (Experimental) RL practice starting from the imitation policy
python train_rl.py --task half --bc policies/half_bc.pt --demos demos/half.npz --out policies/half_sac
```

Results in testing (success on 20 unseen, randomised sheets):

| Fold | Taught steps | Imitation (`train_bc.py`) |
|---|---|---|
| Diagonal | 10/10 | 14-18/20 |
| Half | 9/10 | 17-18/20 |

RL status: starting SAC from the imitation policy works (same success rate), but short
practice runs (4k steps) dropped it to ~55-60% even with the demo regulariser
(`--demo-weight`). Longer runs and tuning are still to do, so use `best_model.zip` and
compare it with the imitation policy before trusting it.

## Swapping the gripper

Everything gripper-specific (joint, open/closed targets, when it counts as holding) is in
the `Gripper` settings in `sim/so101.py`. The policy only sends a grip command in [-1, 1]
and sees an "openness" value in [0, 1], so a different gripper means a new model plus a new
`Gripper(...)` passed to `FoldEnv(gripper=...)`; the training scripts don't change.

## Teaching a new fold

1. Add a task to `TASKS` in `sim/fold_env.py`: which sheet points should end up on top
   of which (this is what the reward and success check measure).
2. Add the matching step list to `STEPS` in `sim/teach.py`
   (`hover`, `down`, `close`, `up`, `drop_on`, `open`, `home`).
3. Run `python teach.py --task <name> --view` until it folds reliably, then record demos.

## Notes

- Grasping is "assisted": closing the gripper near the sheet pins the nearest cloth
  vertex to the jaws. This keeps learning focused on the folding motion.
- Sheet stiffness is `young` in `fold_scene.xml`: lower is floppier (cloth), higher is
  springier. MuJoCo doesn't model permanent creases, so stiff "paper" springs back open.
- Place the sim `laptop` camera where your real webcam will sit, so simulated and real
  images look alike when you move to the real arm.
