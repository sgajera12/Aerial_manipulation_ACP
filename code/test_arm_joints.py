"""
OpenManipulator-X Joint Test Script
Moves each joint one at a time so you can see what each one does.

Controls:
  - The script cycles through each joint automatically with output in terminal

Actuator mapping:
  ctrl[0] = Joint1 (base rotation)
  ctrl[1] = Joint2 (shoulder pitch)
  ctrl[2] = Joint3 (elbow pitch)
  ctrl[3] = Joint4 (wrist pitch)
  ctrl[4] = Gripper (open/close)
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model
model = mujoco.MjModel.from_xml_path("/home/pinaka/robotis_mujoco_menagerie/robotis_open_manipulator_x/scene.xml")
data = mujoco.MjData(model)

# Joint names for printing
joint_names = ["Joint1 (Base)", "Joint2 (Shoulder)", "Joint3 (Elbow)", "Joint4 (Wrist)", "Gripper"]

# Target positions for each joint demo (radians, except gripper in meters)
# Each row: [Joint1, Joint2, Joint3, Joint4, Gripper]
demos = [
    # Move Joint1: rotate base left then right
    {"joint": 0, "targets": [1.0, -1.0, 0.0], "name": joint_names[0]},
    # Move Joint2: shoulder up then down
    {"joint": 1, "targets": [0.8, -0.8, 0.0], "name": joint_names[1]},
    # Move Joint3: elbow up then down
    {"joint": 2, "targets": [0.8, -0.8, 0.0], "name": joint_names[2]},
    # Move Joint4: wrist up then down
    {"joint": 3, "targets": [1.0, -1.0, 0.0], "name": joint_names[3]},
    # Move Gripper: open then close
    {"joint": 4, "targets": [0.019, -0.01, 0.0], "name": joint_names[4]},
]

print("=" * 50)
print("OpenManipulator-X Joint Test")
print("Watch each joint move one at a time!")
print("=" * 50)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Let the robot settle at home position
    print("\n>>> Starting at HOME position...")
    for _ in range(500):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

    # Cycle through each joint
    for demo in demos:
        joint_idx = demo["joint"]
        
        for target in demo["targets"]:
            # Set the target position
            data.ctrl[joint_idx] = target
            print(f"\n>>> {demo['name']} -> target: {target:.3f}")
            
            # Simulate for 2 seconds to let it reach the target
            for _ in range(1000):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
                
                if not viewer.is_running():
                    break
            
            if not viewer.is_running():
                break
        
        # Return this joint to zero before moving next
        data.ctrl[joint_idx] = 0.0
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
        
        if not viewer.is_running():
            break

    print("\n>>> Demo complete! Close the viewer window to exit.")
    
    # Keep viewer open
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
