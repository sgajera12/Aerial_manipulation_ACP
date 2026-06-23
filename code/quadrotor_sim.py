"""
Quadrotor SE(3) Geometric Controller — MuJoCo Implementation based on the paper "Geometric Tracking Control of a Quadrotor UAV on SE(3)"

Usage: python quadrotor_sim.py { --mode fall or hover or control} 

MuJoCo actuator layout (from XML):
    ctrl[0] = body thrust (force along body +z, range [0, 40] N)
    ctrl[1] = Mx (body-frame torque,range [-0.5, 0.5] Nm)
    ctrl[2] = My (body-frame torque,range[-0.5, 0.5] Nm)
    ctrl[3] = Mz (body-frame torque,range[-0.5, 0.5] Nm)

Convention difference from the paper:
    Paper:  z-down, thrust = -f*R*e3 (pushes opposite to b3)
    MuJoCo: z-up,   thrust = +f*R*e3 (pushes along b3)
    This flips some signs in the outer loop. Inner loop (SO(3)) is unchanged.
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
MODEL_PATH = "quadrotor_verify_mass_inertia.xml"

#SO(3) helper functions
def quat_to_rot(q):
    #Each column of R is a body axis expressed in world frame
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def hat(v):
    #3x3 skew-symmetric matrix.
    return np.array([
        [ 0.0, -v[2], v[1]],
        [ v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0 ]
    ])

def vee(M):
    #Inverse of hat. Extracts the 3-vector from a skew-symmetric matrix
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

# SE(3) Geometric Controller
class SE3Controller:
    #Two layers:Outer (position): errors e_x, e_v --> thrust f + desired attitude R_d; Inner (attitude): errors e_R, e_Omega --> moment M

    def __init__(self, mass, inertia_diag, gravity=9.81):
        self.m = mass
        self.J = np.diag(inertia_diag) 
        self.g = gravity
        self.e3 = np.array([0.0, 0.0, 1.0])  # z-up in MuJoCo

        # Position PD (outer loop)
        self.k_x = 6.0 
        self.k_v = 4.0
        # Attitude PD (inner loop)
        self.k_R = 0.4 # proportional
        self.k_Omega = 0.15 # derivative

    def compute(self, x, v, R, Omega,x_d, v_d, x_dd, b1_d,Omega_d=None, Omega_d_dot=None):
        
        if Omega_d is None:
            Omega_d = np.zeros(3)
        if Omega_d_dot is None:
            Omega_d_dot = np.zeros(3)

        m = self.m
        g = self.g
        J = self.J
        e3 = self.e3

        # outer layer - moving to the Position control
        # Position and velocity errors
        e_x = x - x_d
        e_v = v - v_d

        #Desired force vector (adapted for MuJoCo z-up)
        A = -self.k_x * e_x - self.k_v * e_v + m * g * e3 + m * x_dd

        #Thrust magnitude
        f = np.dot(A, R @ e3)

        # Desired body z-axis
        A_norm = np.linalg.norm(A)
        if A_norm < 1e-6:
            b3_d = e3
        else:
            b3_d = A / A_norm

        b3_cross_b1 = np.cross(b3_d, b1_d)
        b3_cross_b1_norm = np.linalg.norm(b3_cross_b1)

        if b3_cross_b1_norm < 1e-6:
            # b1_d is parallel to b3_d —pick an arbitrary perpendicular heading
            if abs(b3_d[0]) < 0.9:
                fallback = np.array([1.0, 0.0, 0.0])
            else:
                fallback = np.array([0.0, 1.0, 0.0])
            b3_cross_b1 = np.cross(b3_d, fallback)
            b3_cross_b1_norm = np.linalg.norm(b3_cross_b1)

        b2_d = b3_cross_b1 / b3_cross_b1_norm
        b1_d_actual = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d_actual, b2_d, b3_d])

        # Inner Layer — Attitude control

        #Attitude error- the drone needs to rotate about to reach R_d.
        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)

        #Angular velocity error
        e_Omega = Omega - R.T @ R_d @ Omega_d

        #Control moment
        feedforward = J @ (hat(Omega) @ R.T @ R_d @ Omega_d- R.T @ R_d @ Omega_d_dot)
        #Final
        M = (-self.k_R * e_R - self.k_Omega * e_Omega + np.cross(Omega, J @ Omega) - feedforward)
        return f, M


# # Simulation runner
# def print_model_info(model):
#     """One-time printout to verify model parameters."""
#     print("=" * 60)
#     print("MODEL INFO")
#     print("=" * 60)
#     drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone_1")
#     mass = model.body_mass[drone_id]
#     inertia = model.body_inertia[drone_id]
#     print(f"Drone body id:{drone_id}")
#     print(f"Drone mass:{mass:.4f} kg")
#     print(f"Drone inertia:[{inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f}] kg·m²")
#     print(f"Hover thrust (mg):{mass * 9.81:.4f} N")
#     print()
#     print("Actuators:")
#     for i in range(model.nu):
#         name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#         lo, hi = model.actuator_ctrlrange[i]
#         print(f"  [{i}] {name:30s}  range [{lo:>6.2f}, {hi:>6.2f}]")
#     print("=" * 60)


def run(mode: str):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    #print_model_info(model)

    # drone physical properties
    drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone_1")
    mass = model.body_mass[drone_id]
    inertia = model.body_inertia[drone_id].copy()
    weight = mass * 9.81

    #Sensor addresses 
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "drone_1_gyro")
    gyro_adr = model.sensor_adr[gyro_id]

    #Controller Setup
    ctrl = None
    x_d = np.array([1.0, 1.0, 3.0])
    v_d = np.zeros(3) 
    x_dd = np.zeros(3)
    b1_d = np.array([1.0, 0.0, 0.0])

    if mode == "control":
        ctrl = SE3Controller(mass, inertia)
        print(f"\nController gains:")
        print(f"Position: k_x = {ctrl.k_x},k_v = {ctrl.k_v}")
        print(f"Attitude: k_R = {ctrl.k_R},k_Omega = {ctrl.k_Omega}")
        print(f"Target: x_d = {x_d}")
        print(f"Heading: b1_d = {b1_d}")

    print(f"\nMode: {mode}")
    print("Close the viewer window to stop.\n")

    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # CONTROL MODE
            if mode == "fall":
                data.ctrl[:] = 0.0

            elif mode == "hover":
                data.ctrl[0] = weight
                data.ctrl[1:4] = 0.0

            #read from mujoco
            elif mode == "control":
                x= data.qpos[0:3].copy()
                quat = data.qpos[3:7].copy()
                v= data.qvel[0:3].copy()
                Omega = data.sensordata[gyro_adr:gyro_adr+3].copy()
                R= quat_to_rot(quat)

                f, M = ctrl.compute(x, v, R, Omega,x_d, v_d, x_dd, b1_d)

                # Clamp to actuator limits and send
                data.ctrl[0] = np.clip(f,0.0,  40.0)# thrust
                data.ctrl[1] = np.clip(M[0], -0.5, 0.5)# Mx
                data.ctrl[2] = np.clip(M[1], -0.5,0.5)# My
                data.ctrl[3] = np.clip(M[2], -0.5,0.5)# Mz

            # STEP 
            mujoco.mj_step(model, data)
            viewer.sync()

            # Print state periodically 
            if data.time - last_print > 0.5:
                pos = data.qpos[:3]
                if mode == "control":
                    err = np.linalg.norm(pos - x_d)
                    print(f"t={data.time:6.2f}"
                          f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}]"
                          f"|e_x|={err:.4f}"
                          f"ctrl=[{data.ctrl[0]:5.2f},{data.ctrl[1]:+.3f},{data.ctrl[2]:+.3f},{data.ctrl[3]:+.3f}]")
                else:
                    print(f"t={data.time:6.2f}"
                          f"pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}]")
                last_print = data.time

            #Real-time pacing 
            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fall", "hover", "control"],default="fall")
    args = parser.parse_args()
    run(args.mode)
