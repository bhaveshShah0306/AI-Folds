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
