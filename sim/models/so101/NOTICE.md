# SO-101 model

`so101_new_calib_camera.xml` and everything in `assets/` come from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(`Simulation/SO101`), licensed under Apache-2.0 (see `LICENSE`).

Changes made for AI-Folds:
- Added a `wrist` MuJoCo camera inside the `wrist_camera` body.
- `fold_scene.xml` is new: it adds the floor, a cloth sheet, an assisted-grasp
  constraint and the `laptop` / `overhead` cameras.
