"""
Quadrotor MuJoCo simulation viewer.

Usage:
    python quadrotor_sim.py --mode fall      # no control, just gravity
    python quadrotor_sim.py --mode hover     # constant thrust = weight (open-loop)
    python quadrotor_sim.py --mode control   # geometric controller (filled in later)

The XML exposes 4 actuators in this order:
    0: body_thrust   (force along body +z,  range 0..40 N)
    1: x_moment      (body-frame torque,    range -0.5..0.5 N·m)
    2: y_moment      (body-frame torque,    range -0.5..0.5 N·m)
    3: z_moment      (body-frame torque,    range -0.5..0.5 N·m)

So we directly command [f, Mx, My, Mz] — no rotor mixing needed.
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH = "/home/pinaka/robotis_mujoco_menagerie/drone/quadrotor_verify_mass_inertia.xml"


def print_model_info(model, data):
    """One-time printout so we can verify the model matches what we expect."""
    print("=" * 60)
    print("MODEL INFO")
    print("=" * 60)
    print(f"Bodies:       {model.nbody}")
    print(f"DOF (nv):     {model.nv}")
    print(f"Actuators:    {model.nu}")
    print(f"Timestep:     {model.opt.timestep}")
    print(f"Gravity:      {model.opt.gravity}")
    print()

    # The drone body is index 1 (0 is world). Get its mass from the inertia.
    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone_1")
    print(f"Drone body id:        {drone_body_id}")
    print(f"Drone mass (body):    {model.body_mass[drone_body_id]:.4f} kg")
    print(f"Drone inertia diag:   {model.body_inertia[drone_body_id]}")
    print(f"Drone weight:         {model.body_mass[drone_body_id] * 9.81:.4f} N")
    print()

    print("Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  [{i}] {name:30s}  range [{lo:>6.2f}, {hi:>6.2f}]")
    print("=" * 60)


def run(mode: str):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print_model_info(model, data)

    # Drone body for printing state during sim
    drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone_1")
    mass = model.body_mass[drone_id]
    weight = mass * 9.81  # exact hover thrust

    print(f"\nRunning in mode: {mode}")
    if mode == "hover":
        print(f"Sending constant body thrust = {weight:.4f} N (= weight)")
    print("Close the viewer window to stop.\n")

    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # ---- control law goes here ----
            if mode == "fall":
                data.ctrl[:] = 0.0
            elif mode == "hover":
                data.ctrl[0] = weight   # body thrust
                data.ctrl[1] = 0.0      # Mx
                data.ctrl[2] = 0.0      # My
                data.ctrl[3] = 0.0      # Mz
            elif mode == "control":
                # placeholder — geometric controller goes here next
                data.ctrl[:] = 0.0
                data.ctrl[0] = weight
            else:
                raise ValueError(f"unknown mode: {mode}")
            # -------------------------------

            mujoco.mj_step(model, data)
            viewer.sync()

            # print drone z every 0.5 s so you can see what's happening
            if data.time - last_print > 0.5:
                pos = data.qpos[:3]
                quat = data.qpos[3:7]
                print(f"t={data.time:6.2f}s   pos=[{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]   "
                      f"quat=[{quat[0]:+.2f},{quat[1]:+.2f},{quat[2]:+.2f},{quat[3]:+.2f}]")
                last_print = data.time

            # real-time pacing
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fall", "hover", "control"], default="fall")
    args = parser.parse_args()
    run(args.mode)
