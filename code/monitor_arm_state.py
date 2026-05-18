"""
OpenManipulator-X State Monitoring Script
Commands the arm to a pose and prints live state data.

This teaches you how to:
  1. Read joint positions (data.qpos)
  2. Read joint velocities (data.qvel)
  3. Read actuator forces (data.qfrc_actuator)
  4. Compute tracking error (commanded vs actual)
  5. Access end-effector position (forward kinematics)

Press Ctrl+C in terminal to stop.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Load model ---
model = mujoco.MjModel.from_xml_path(
    "/home/pinaka/robotis_mujoco_menagerie/robotis_open_manipulator_x/scene.xml"
)
data = mujoco.MjData(model)

# --- Understand the model structure ---
print("=" * 60)
print("MODEL INFO")
print("=" * 60)
print(f"Number of joints (nq):      {model.nq}")
print(f"Number of DOF (nv):         {model.nv}")
print(f"Number of actuators (nu):   {model.nu}")
print(f"Number of bodies:           {model.nbody}")
print(f"Simulation timestep:        {model.opt.timestep} s")
print(f"Gravity:                    {model.opt.gravity}")

# Print joint names and their indices
print("\n--- Joint Names ---")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]  # 0=free, 1=ball, 2=slide, 3=hinge
    type_str = ["free", "ball", "slide", "hinge"][jnt_type]
    print(f"  Joint {i}: {name:20s} type={type_str}")

# Print actuator names
print("\n--- Actuator Names ---")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  Actuator {i}: {name}")

# Print body names (useful for end-effector tracking)
print("\n--- Body Names ---")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  Body {i}: {name}")

# --- Get body ID for end-effector ---
# link5 is the last link before the gripper
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
print(f"\nEnd-effector body ID: {ee_body_id} (link5)")

# --- Define target pose ---
# A pick-ready pose: base rotated, arm reaching forward and down
target_ctrl = np.array([
    0.5,    # Joint1: base rotated 0.5 rad (~29 degrees)
    -0.3,   # Joint2: shoulder slightly up
    0.4,    # Joint3: elbow bent
    0.8,    # Joint4: wrist angled down
    0.019   # Gripper: open
])

print("\n" + "=" * 60)
print("COMMANDING ARM TO TARGET POSE")
print("=" * 60)
print(f"Target: {target_ctrl}")
print("\nStarting simulation... (Ctrl+C to stop)\n")

# --- Simulation loop ---
step_count = 0
print_interval = 500  # print every 500 steps (every 1 second)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set the target
    data.ctrl[:] = target_ctrl
    
    try:
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
            step_count += 1
            
            # Print state every 1 second
            if step_count % print_interval == 0:
                sim_time = data.time
                
                # Current joint positions (what the robot actually is)
                qpos = data.qpos[:5]  # first 5 (ignore gripper mimic)
                
                # Current joint velocities
                qvel = data.qvel[:5]
                
                # Tracking error = commanded - actual
                error = target_ctrl.copy()
                error[:4] -= qpos[:4]           # joints 1-4 (radians)
                error[4] -= qpos[4]             # gripper (meters)
                
                # Actuator forces being applied
                actuator_force = data.qfrc_actuator[:5]
                
                # End-effector position (forward kinematics - MuJoCo computes this for us)
                ee_pos = data.xpos[ee_body_id]  # 3D position in world frame
                
                # Print everything
                print(f"--- t = {sim_time:.1f}s ---")
                print(f"  Joint pos (rad):   [{', '.join(f'{q:+.4f}' for q in qpos)}]")
                print(f"  Joint vel (rad/s): [{', '.join(f'{v:+.4f}' for v in qvel)}]")
                print(f"  Tracking error:    [{', '.join(f'{e:+.4f}' for e in error)}]")
                print(f"  Actuator force:    [{', '.join(f'{f:+.4f}' for f in actuator_force)}]")
                print(f"  End-effector XYZ:  [{', '.join(f'{p:+.4f}' for p in ee_pos)}]")
                print()
                
                # After settling, change pose to show dynamic response
                if step_count == 3000:  # at 6 seconds
                    print(">>> CHANGING TO NEW POSE!")
                    target_ctrl_new = np.array([-0.5, 0.3, -0.4, -0.5, -0.01])
                    data.ctrl[:] = target_ctrl_new
                    target_ctrl[:] = target_ctrl_new
                    print(f">>> New target: {target_ctrl}\n")
                    
    except KeyboardInterrupt:
        print("\nStopped by user.")

print("Done.")
