"""
Computed Torque Controller for OpenManipulator-X in MuJoCo
==========================================================

Same control law as RBE 502 final project:
    τ = M(q)(q̈_d + Kv*ė + Kp*e) + C(q,q̇)q̇ + G(q)

where:
    e  = q_d - q        (position error)
    ė  = q̇_d - q̇       (velocity error)
    M  = inertia matrix
    C  = Coriolis/centrifugal matrix  
    G  = gravity vector

The KEY difference from RBE 502:
    - In RBE 502, you computed M, C, G yourself from DH parameters
    - In MuJoCo, the engine computes these for us via mj_fullM() and 
      the bias force vector (which contains C*q̇ + G)

This makes MuJoCo ideal for testing controllers — you get perfect 
dynamics computation, and can focus on the control design.

Tasks:
    1. Pose regulation (go to a fixed target)
    2. Trajectory tracking (follow a sinusoidal trajectory)
    
Press Ctrl+C to stop.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ComputedTorqueController:
    """
    Computed Torque Controller
    
    τ = M(q)(q̈_d + Kv*ė + Kp*e) + C(q,q̇)q̇ + G(q)
    
    In MuJoCo terms:
    - M comes from mj_fullM()
    - C*q̇ + G comes from data.qfrc_bias (the bias force)
    """
    
    def __init__(self, model, Kp, Kv):
        """
        Args:
            model: MuJoCo model
            Kp: Proportional gain matrix (nv x nv)
            Kv: Derivative gain matrix (nv x nv)
        """
        self.model = model
        self.nv = model.nv  # number of DOF (degrees of freedom)
        self.Kp = Kp
        self.Kv = Kv
        # Pre-allocate mass matrix (full dense matrix)
        self.M = np.zeros((self.nv, self.nv))
    
    def get_mass_matrix(self, data):
        """Extract the mass matrix M(q) from MuJoCo."""
        # mj_fullM fills a dense mass matrix
        mujoco.mj_fullM(self.model, self.M, data.qM)
        return self.M.copy()
    
    def get_bias_forces(self, data):
        """
        Get C(q,q̇)q̇ + G(q) from MuJoCo.
        
        data.qfrc_bias contains the total bias force:
        this is exactly C(q,q̇)*q̇ + G(q), which is what we need
        for the computed torque law.
        """
        return data.qfrc_bias.copy()
    
    def compute_torque(self, data, q_d, qd_d, qdd_d):
        """
        Compute control torques.
        
        Args:
            data: MuJoCo data (contains current state)
            q_d: desired joint positions (nv,)
            qd_d: desired joint velocities (nv,)
            qdd_d: desired joint accelerations (nv,)
            
        Returns:
            tau: joint torques (nv,)
        """
        # Current state
        q = data.qpos[:self.nv]      # current positions
        qd = data.qvel[:self.nv]     # current velocities
        
        # Errors
        e = q_d - q                  # position error
        ed = qd_d - qd              # velocity error
        
        # Get dynamics from MuJoCo
        M = self.get_mass_matrix(data)
        bias = self.get_bias_forces(data)  # C*q̇ + G
        
        # Computed torque law:
        # τ = M(q̈_d + Kv*ė + Kp*e) + C*q̇ + G
        aq = qdd_d + self.Kv @ ed + self.Kp @ e
        tau = M @ aq + bias
        
        return tau, e, ed


def run_pose_regulation(controller, model, data, q_target, duration=10.0):
    """
    Task 1: Pose Regulation
    Move from current position to a fixed target.
    Same as RBE 502 pose regulation task.
    """
    print("\n" + "=" * 60)
    print("TASK 1: POSE REGULATION (Computed Torque)")
    print("=" * 60)
    print(f"Target: {q_target}")
    
    nv = model.nv
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    
    # Desired: fixed position, zero velocity & acceleration
    q_d = q_target.copy()
    qd_d = np.zeros(nv)
    qdd_d = np.zeros(nv)
    
    # Logging arrays
    times = []
    positions = []
    errors = []
    torques_log = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break
            
            # Compute torques
            tau, e, ed = controller.compute_torque(data, q_d, qd_d, qdd_d)
            
            # Apply torques (only first 4 joints, gripper gets 0)
            data.ctrl[:4] = tau[:4]
            data.ctrl[4] = 0.0
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
            
            # Log data
            times.append(data.time)
            positions.append(data.qpos[:4].copy())
            errors.append(e[:4].copy())
            torques_log.append(tau[:4].copy())
            
            # Print every 2 seconds
            if i % 1000 == 0:
                err_norm = np.linalg.norm(e[:4])
                print(f"  t={data.time:5.1f}s | error norm: {err_norm:.6f} rad")
    
    return np.array(times), np.array(positions), np.array(errors), np.array(torques_log)


def run_trajectory_tracking(controller, model, data, duration=20.0):
    """
    Task 2: Trajectory Tracking
    Follow a sinusoidal trajectory.
    Same as RBE 502 trajectory tracking task.
    """
    print("\n" + "=" * 60)
    print("TASK 2: TRAJECTORY TRACKING (Computed Torque)")
    print("=" * 60)
    
    nv = model.nv
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    
    # Sinusoidal trajectory parameters (similar to RBE 502)
    omega = 0.3  # rad/s
    amplitudes = np.array([0.15, 0.10, 0.08, 0.05, 0.0, 0.0])  # last 2 for gripper DOFs
    midpoints = np.array([0.30, -0.20, 0.25, 0.10, 0.0, 0.0])
    
    print(f"Frequency: {omega} rad/s")
    print(f"Amplitudes: {amplitudes[:4]}")
    print(f"Midpoints:  {midpoints[:4]}")
    
    # Logging arrays
    times = []
    positions = []
    desired_positions = []
    errors = []
    torques_log = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break
            
            t = data.time
            
            # Desired trajectory: q_d = midpoint + amplitude * sin(omega * t)
            q_d = midpoints + amplitudes * np.sin(omega * t)
            qd_d = amplitudes * omega * np.cos(omega * t)
            qdd_d = -amplitudes * omega**2 * np.sin(omega * t)
            
            # Compute torques
            tau, e, ed = controller.compute_torque(data, q_d, qd_d, qdd_d)
            
            # Apply torques
            data.ctrl[:4] = tau[:4]
            data.ctrl[4] = 0.0
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
            
            # Log data
            times.append(t)
            positions.append(data.qpos[:4].copy())
            desired_positions.append(q_d[:4].copy())
            errors.append(e[:4].copy())
            torques_log.append(tau[:4].copy())
            
            # Print every 2 seconds
            if i % 1000 == 0:
                err_norm = np.linalg.norm(e[:4])
                print(f"  t={t:5.1f}s | error norm: {err_norm:.6f} rad")
    
    return (np.array(times), np.array(positions), np.array(desired_positions),
            np.array(errors), np.array(torques_log))


def plot_pose_regulation(times, positions, errors, torques, q_target):
    """Plot pose regulation results — same style as RBE 502 Fig. 1"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pose Regulation — Computed Torque Controller", fontsize=14)
    joint_labels = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]
    
    # Joint positions
    ax = axes[0, 0]
    for j in range(4):
        ax.plot(times, positions[:, j], label=joint_labels[j])
        ax.axhline(y=q_target[j], color=f'C{j}', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Positions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tracking errors
    ax = axes[0, 1]
    for j in range(4):
        ax.plot(times, errors[:, j], label=joint_labels[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (rad)")
    ax.set_title("Tracking Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Torques
    ax = axes[1, 0]
    for j in range(4):
        ax.plot(times, torques[:, j], label=joint_labels[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Joint Torques")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error norm
    ax = axes[1, 1]
    err_norm = np.linalg.norm(errors, axis=1)
    ax.plot(times, err_norm, 'k-', linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||e|| (rad)")
    ax.set_title("Norm of Tracking Error")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pose_regulation_CT.png", dpi=150)
    print("Saved: pose_regulation_CT.png")
    plt.show()


def plot_trajectory_tracking(times, positions, desired, errors, torques):
    """Plot trajectory tracking results — same style as RBE 502 Fig. 3"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trajectory Tracking — Computed Torque Controller", fontsize=14)
    joint_labels = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]
    
    # Joint positions vs desired
    ax = axes[0, 0]
    for j in range(4):
        ax.plot(times, positions[:, j], label=f"{joint_labels[j]} actual")
        ax.plot(times, desired[:, j], '--', alpha=0.5, label=f"{joint_labels[j]} desired")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Positions")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Tracking errors
    ax = axes[0, 1]
    for j in range(4):
        ax.plot(times, errors[:, j], label=joint_labels[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (rad)")
    ax.set_title("Tracking Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Torques
    ax = axes[1, 0]
    for j in range(4):
        ax.plot(times, torques[:, j], label=joint_labels[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Joint Torques")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error norm
    ax = axes[1, 1]
    err_norm = np.linalg.norm(errors, axis=1)
    ax.plot(times, err_norm, 'k-', linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||e|| (rad)")
    ax.set_title("Norm of Tracking Error")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trajectory_tracking_CT.png", dpi=150)
    print("Saved: trajectory_tracking_CT.png")
    plt.show()


# MAIN
if __name__ == "__main__":
    # Load torque-controlled model
    model_path = "/home/pinaka/robotis_mujoco_menagerie/robotis_open_manipulator_x/scene_torque.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    nv = model.nv  # degrees of freedom
    print(f"Degrees of freedom: {nv}")
    
    # --- Controller Gains ---
    # Same structure as RBE 502: Kp and Kv are diagonal matrices
    # Tuned for MuJoCo simulation (different from hardware gains)
    Kp = np.diag([100.0, 200.0, 200.0, 50.0, 10.0, 10.0])  # include gripper DOFs
    Kv = np.diag([20.0, 40.0, 40.0, 10.0, 5.0, 5.0])
    
    controller = ComputedTorqueController(model, Kp, Kv)
    
    # Task 1: Pose Regulation
    # Reset to home
    mujoco.mj_resetData(model, data)
    
    # Target pose (same as RBE 502)
    q_target = np.array([0.50, -0.35, 0.20, 0.50, 0.0, 0.0])
    
    print("\nClose the viewer window when you're done watching to proceed to plotting.")
    times, positions, errors, torques = run_pose_regulation(
        controller, model, data, q_target, duration=3.0
    )
    
    # Plot results
    plot_pose_regulation(times, positions, errors, torques, q_target[:4])
    
    #Task 2: Trajectory Tracking
    input("\nPress Enter to start trajectory tracking...")
    
    # Reset
    mujoco.mj_resetData(model, data)
    # Start near the trajectory midpoint for smoother start
    data.qpos[:4] = [0.30, -0.20, 0.25, 0.10]
    mujoco.mj_forward(model, data)
    
    times, positions, desired, errors, torques = run_trajectory_tracking(
        controller, model, data, duration=10.0
    )
    
    # Plot results
    plot_trajectory_tracking(times, positions, desired, errors, torques)
    
    print("\nDone! Check the saved plots.")
