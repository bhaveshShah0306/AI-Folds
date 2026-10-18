"""SO-101 helpers: load the MuJoCo model, read the arm's "landmarks", and solve IK.

Landmarks are the arm's equivalent of MediaPipe hand keypoints: the world-space
position of every joint axis plus the gripper tip, recomputed each sim step.
"""

from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).parent / "models" / "so101"
ARM_XML = MODEL_DIR / "so101_new_calib_camera.xml"
FOLD_SCENE_XML = MODEL_DIR / "fold_scene.xml"

# Joint order matches the actuators and LeRobot's SO-101 motor order.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
TIP_SITE = "gripperframe"

# Gripper joint targets (radians). The jaw opens towards the upper limit.
GRIPPER_OPEN = 1.2
GRIPPER_CLOSED = 0.0


def landmarks(model, data):
    """Return {name: xyz} for every joint anchor plus the gripper tip."""
    points = {name: data.xanchor[model.joint(name).id].copy() for name in JOINTS}
    points["tip"] = data.site(TIP_SITE).xpos.copy()
    return points


def landmark_array(model, data):
    """Landmarks as a (7, 3) array, in JOINTS order followed by the tip."""
    return np.array(list(landmarks(model, data).values()))


class IK:
    """Damped least-squares IK for the gripper tip.

    Solves for tip position and, optionally, for the gripper pointing straight
    down (the tip site's local x axis is the approach direction). Position plus
    pointing direction is 5 constraints, which the 5 arm joints can satisfy.
    """

    def __init__(self, model, damping=1e-3, max_iters=100, tol=1e-4):
        self.model = model
        self.data = mujoco.MjData(model)
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        self.site_id = model.site(TIP_SITE).id
        self.qpos_idx = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
        self.dof_idx = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
        self.lo = np.array([model.jnt_range[model.joint(j).id][0] for j in ARM_JOINTS])
        self.hi = np.array([model.jnt_range[model.joint(j).id][1] for j in ARM_JOINTS])

    def solve(self, target_pos, q_init, point_down=True):
        """Return (arm joint angles, final position error in metres)."""
        model, data = self.model, self.data
        q = np.clip(np.asarray(q_init, dtype=float).copy(), self.lo, self.hi)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        down = np.array([0.0, 0.0, -1.0])
        err_pos = np.inf
        for _ in range(self.max_iters):
            data.qpos[self.qpos_idx] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            err_pos = target_pos - data.site_xpos[self.site_id]
            mujoco.mj_jacSite(model, data, jacp, jacr, self.site_id)
            J = jacp[:, self.dof_idx]
            err = err_pos
            if point_down:
                approach = data.site_xmat[self.site_id].reshape(3, 3)[:, 0]
                # Rotation needed to swing the approach axis onto straight down.
                err_rot = np.cross(approach, down)
                J = np.vstack([J, 0.3 * jacr[:, self.dof_idx]])
                err = np.concatenate([err_pos, 0.3 * err_rot])
            if np.linalg.norm(err) < self.tol:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(J.shape[0]), err)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, float(np.linalg.norm(err_pos))
