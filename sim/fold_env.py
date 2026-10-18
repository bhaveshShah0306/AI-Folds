"""Gymnasium environment: an SO-101 arm folding a square sheet in MuJoCo.

Action (4 floats in [-1, 1]):
    dx, dy, dz  move the gripper tip by up to MAX_STEP metres per step
    grip        > 0 closes the gripper, <= 0 opens it
The gripper always points straight down; IK turns the tip target into joint angles.
Gripper details (joint, angles, when it counts as holding) come from a `Gripper`
in so101.py, so the policy only deals with a generic grip command and openness.

Grasping is "assisted": once the gripper has closed within GRASP_RADIUS of the sheet,
the nearest sheet vertex is pinned to it, and it's let go once the gripper has
opened again. Real cloth grasping in simulation is very fiddly, and this is the
usual shortcut for learning the folding motion itself.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101 import ARM_JOINTS, FOLD_SCENE_XML, IK, SO101_JAW, TIP_SITE, landmark_array

GRID = 9                 # sheet is GRID x GRID vertices
MAX_STEP = 0.01          # metres per env step
GRASP_RADIUS = 0.025
SUCCESS_DIST = 0.02
CONTROL_SUBSTEPS = 25    # 25 x 2 ms = 20 Hz control
WORKSPACE_LO = np.array([0.10, -0.15, 0.005])
WORKSPACE_HI = np.array([0.32, 0.15, 0.15])
HOME_TIP = np.array([0.20, 0.0, 0.10])
# Observations are scaled to roughly [-1, 1] so learners see centimetre-level
# differences clearly without having to normalise inputs themselves.
WS_CENTRE = (WORKSPACE_LO + WORKSPACE_HI) / 2
WS_HALF = (WORKSPACE_HI - WORKSPACE_LO) / 2
REL_SCALE = 0.1


def vertex(ix, iy):
    """Sheet vertex index from grid coordinates. ix runs away from the robot, iy to its left."""
    return ix * GRID + iy


# Named points on the sheet. "Right"/"left" are from the robot's point of view.
POINTS = {
    "near_right": vertex(0, 0),
    "near_left": vertex(0, GRID - 1),
    "far_right": vertex(GRID - 1, 0),
    "far_left": vertex(GRID - 1, GRID - 1),
    "near_mid": vertex(0, GRID // 2),
    "far_mid": vertex(GRID - 1, GRID // 2),
    "centre": vertex(GRID // 2, GRID // 2),
}

# A task is a list of (moving point, target point) pairs that should end up on top
# of each other. Add your own to teach other folds.
TASKS = {
    "diagonal": [("far_left", "near_right")],
    "half": [("far_right", "near_right"), ("far_left", "near_left"), ("far_mid", "near_mid")],
}


class FoldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task="diagonal", max_steps=200, randomize=True, render_mode=None, gripper=SO101_JAW):
        self.model = mujoco.MjModel.from_xml_path(str(FOLD_SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.ik = IK(self.model)
        self.task_name = task
        self.pairs = [(POINTS[a], POINTS[b]) for a, b in TASKS[task]]
        self.max_steps = max_steps
        self.randomize = randomize
        self.render_mode = render_mode
        self.gripper = gripper
        self._renderer = None

        m = self.model
        self.vertex_bodies = np.array([m.body(f"sheet_{i}").id for i in range(GRID * GRID)])
        self.vertex_qpos = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.vertex_bodies])
        self.rest_pos = m.body_pos[self.vertex_bodies].copy()
        self.arm_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.gripper_qpos = m.jnt_qposadr[m.joint(gripper.joint).id]
        self.gripper_ctrl = m.actuator(gripper.joint).id
        self.grasp_eq = m.equality("grasp").id
        self.gripper_body = m.body("gripper").id
        self.tip_site = m.site(TIP_SITE).id
        self.corners = np.array([POINTS[k] for k in ("near_right", "near_left", "far_right", "far_left")])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # arm joints, gripper openness, tip, holding flag, corners, corners relative
        # to the tip, sheet centre
        obs_dim = 5 + 1 + 3 + 1 + 4 * 3 + 4 * 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ----------------------------------------------------------------- helpers

    def sheet_positions(self):
        return self.data.xpos[self.vertex_bodies]

    def tip_position(self):
        return self.data.site_xpos[self.tip_site].copy()

    @property
    def grasped_vertex(self):
        if not self.data.eq_active[self.grasp_eq]:
            return None
        return int(np.where(self.vertex_bodies == self.model.eq_obj2id[self.grasp_eq])[0][0])

    def openness(self):
        return self.gripper.openness(self.data.qpos[self.gripper_qpos])

    def _set_grasp(self, close):
        openness = self.openness()
        if close and openness < self.gripper.hold_below and self.grasped_vertex is None:
            tip = self.tip_position()
            dists = np.linalg.norm(self.sheet_positions() - tip, axis=1)
            i = int(np.argmin(dists))
            if dists[i] < GRASP_RADIUS:
                body = self.vertex_bodies[i]
                # Anchor the vertex where it currently sits in the gripper's frame,
                # so grabbing it doesn't make the sheet jump.
                rot = self.data.xmat[self.gripper_body].reshape(3, 3)
                local = rot.T @ (self.data.xpos[body] - self.data.xpos[self.gripper_body])
                self.model.eq_obj2id[self.grasp_eq] = body
                self.model.eq_data[self.grasp_eq, :3] = local
                self.model.eq_data[self.grasp_eq, 3:6] = 0.0
                self.data.eq_active[self.grasp_eq] = 1
        elif not close and openness > self.gripper.release_above:
            self.data.eq_active[self.grasp_eq] = 0

    def fold_distances(self):
        pos = self.sheet_positions()
        return np.array([np.linalg.norm(pos[a] - pos[b]) for a, b in self.pairs])

    def _obs(self):
        pos = self.sheet_positions()
        tip = self.tip_position()
        corners = pos[self.corners]
        scale = lambda p: (p - WS_CENTRE) / WS_HALF
        return np.concatenate([
            self.data.qpos[self.arm_qpos],
            [self.openness()],
            scale(tip),
            [1.0 if self.grasped_vertex is not None else 0.0],
            scale(corners).ravel(),
            ((corners - tip) / REL_SCALE).ravel(),
            scale(pos.mean(axis=0)),
        ]).astype(np.float32)

    def landmarks(self):
        """(7, 3) array: world position of each joint plus the gripper tip."""
        return landmark_array(self.model, self.data)

    # --------------------------------------------------------------- gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.eq_active[self.grasp_eq] = 0

        if self.randomize:
            # Shift and rotate the sheet a little so the policy can't memorise one layout.
            angle = self.np_random.uniform(-0.25, 0.25)
            shift = self.np_random.uniform([-0.015, -0.02, 0], [0.015, 0.02, 0])
            centre = self.rest_pos.mean(axis=0)
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            new_pos = (self.rest_pos - centre) @ rot.T + centre + shift
            for k in range(3):
                self.data.qpos[self.vertex_qpos + k] = new_pos[:, k] - self.rest_pos[:, k]

        self.target = HOME_TIP.copy()
        q, _ = self.ik.solve(self.target, np.zeros(5))
        self.data.qpos[self.arm_qpos] = q
        self.data.qpos[self.gripper_qpos] = self.gripper.open_target
        self.data.ctrl[:5] = q
        self.data.ctrl[self.gripper_ctrl] = self.gripper.open_target
        self.q_target = q
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):  # let the sheet settle
            mujoco.mj_step(self.model, self.data)

        self.steps = 0
        self.prev_dist = self.fold_distances().sum()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.target = np.clip(self.target + action[:3] * MAX_STEP, WORKSPACE_LO, WORKSPACE_HI)
        self.q_target, _ = self.ik.solve(self.target, self.q_target)
        close = action[3] > 0
        self.data.ctrl[:5] = self.q_target
        self.data.ctrl[self.gripper_ctrl] = self.gripper.target(action[3])

        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self._set_grasp(close)
        self.steps += 1

        dists = self.fold_distances()
        total = dists.sum()
        # Reward progress on the fold, plus a nudge towards the sheet while not holding it.
        reward = 10.0 * (self.prev_dist - total)
        self.prev_dist = total
        if self.grasped_vertex is None:
            moving = self.sheet_positions()[[a for a, _ in self.pairs]]
            reward -= 0.1 * np.min(np.linalg.norm(moving - self.tip_position(), axis=1))

        success = bool(np.all(dists < SUCCESS_DIST) and self.grasped_vertex is None)
        unstable = not np.all(np.isfinite(self.data.qpos))
        if success:
            reward += 10.0
        if unstable:
            reward -= 10.0
        terminated = success or unstable
        truncated = self.steps >= self.max_steps
        info = {"fold_distances": dists, "is_success": success, "landmarks": self.landmarks()}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera="front")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
