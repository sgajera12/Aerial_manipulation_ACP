"""
Velocity Control for OpenManipulator-X in MuJoCo
=================================================

Architecture:
    Your controller (outer loop) computes desired velocity:
        q̇_cmd = Kp * (q_d - q)
    
    MuJoCo velocity actuator (inner loop) tracks that velocity:
        τ = kv * (q̇_cmd - q̇)      ← handled by MuJoCo internally

This matches the real Dynamixel servo in velocity mode.

On the real OMX:
    - You send velocity commands via the Dynamixel SDK
    - The servo's internal controller handles torque generation
    - Same architecture, just different hardware

Tasks:
    1. Pose regulation: P controller → velocity command
    2. Trajectory tracking: PI controller + feedforward → velocity command
    
Press Ctrl+C to stop.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class VelocityController:
    """
    For pose regulation (P control): q̇_cmd = Kp * e
    For trajectory tracking (feedforward + PI): q̇_cmd = q̇_d + Kp * e + Ki * ∫e dt
    The velocity command is sent to MuJoCo's velocity actuator, which internally applies: τ = kv * (q̇_cmd - q̇) 
    """
    
    def __init__(self, n_joints, Kp, Ki=None, vel_limit=2.0):
        """Args:
            n_joints: number of controlled joints
            Kp: proportional gain (array or scalar)
            Ki: integral gain (array or scalar), None = no integral
            vel_limit: max velocity command in rad/s
        """
        self.n_joints = n_joints
        self.Kp = np.atleast_1d(Kp) * np.ones(n_joints) #this is the part where we add the 
        self.Ki = np.atleast_1d(Ki) * np.ones(n_joints) if Ki is not None else None
        self.vel_limit = vel_limit
        
        # Integral term accumulator
        self.error_integral = np.zeros(n_joints)
    
    def reset(self):
        """Reset integral term."""
        self.error_integral = np.zeros(self.n_joints)
    
    def compute_velocity(self, q, q_d, qd_d=None, dt=0.002):
        """
        Compute velocity command.
        Args:
            q: current joint positions
            q_d: desired joint positions
            qd_d: desired joint velocities (feedforward), None for regulation
            dt: timestep for integral
        Returns:
            vel_cmd: velocity command (rad/s)
            e: position error
        """
        # Position error
        e = q_d - q
        
        # P term
        vel_cmd = self.Kp * e
        
        # Feedforward (for trajectory tracking)
        if qd_d is not None:
            vel_cmd += qd_d# 
        
        # Integral term
        if self.Ki is not None:
            self.error_integral += e * dt # change in error over time = ErrorIntegral
            # Anti-windup: clamp integral
            self.error_integral = np.clip(self.error_integral, -1.0, 1.0) # to prevent high error over time, if the robot is stuck 
            
            vel_cmd += self.Ki * self.error_integral # Kp * error
        
        # Clamp velocity to motor limits
        vel_cmd = np.clip(vel_cmd, -self.vel_limit, self.vel_limit)
        
        return vel_cmd, e


def run_pose_regulation(model, data, controller, q_target, duration=10.0):
    """Pose regulation using velocity control."""
    print("TASK POSE REGULATION (Velocity Control)")
    print(f"Target: {q_target[:4]}")
    
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    controller.reset()
    
    # Logging
    times, positions, errors, vel_cmds = [], [], [], []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break
            
            # Current joint positions
            q = data.qpos[:4]
            
            # Compute velocity command (P control, no feedforward)
            vel_cmd, e = controller.compute_velocity(q, q_target[:4], dt=dt)
            
            # Send velocity commands to MuJoCo
            data.ctrl[:4] = vel_cmd
            data.ctrl[4] = 0.0  # gripper as its our last motor
            
            # Step
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
            
            # Log
            times.append(data.time)
            positions.append(q.copy())
            errors.append(e.copy())
            vel_cmds.append(vel_cmd.copy())
            
            if i % 1000 == 0:
                print(f"  t={data.time:5.1f}s | error norm: {np.linalg.norm(e):.6f} rad | "
                      f"vel cmd: [{', '.join(f'{v:.3f}' for v in vel_cmd)}]")
    
    return np.array(times), np.array(positions), np.array(errors), np.array(vel_cmds)


def run_trajectory_tracking(model, data, controller, duration=20.0):
    """Trajectory tracking using velocity control with feedforward."""
    print("2 Trajectory Tracking (Velocity Control)")
    
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    controller.reset()
    
    # Same sinusoidal trajectory as
    omega = 0.3
    amplitudes = np.array([0.15, 0.10, 0.08, 0.05])
    midpoints = np.array([0.30, -0.20, 0.25, 0.10])
    
    print(f"Frequency:{omega} rad/s")
    print(f"Amplitudes:{amplitudes}")
    print(f"Midpoints:{midpoints}")
    
    # Logging
    times, positions, desired_pos, errors, vel_cmds = [], [], [], [], []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break
            
            t = data.time
            
            # Desired trajectory
            q_d = midpoints + amplitudes * np.sin(omega * t)
            qd_d = amplitudes * omega * np.cos(omega * t)
            
            # Current state
            q = data.qpos[:4]
            
            # Compute velocity command (feedforward + P + I)
            vel_cmd, e = controller.compute_velocity(q, q_d, qd_d=qd_d, dt=dt)
            
            # Send velocity commands
            data.ctrl[:4] = vel_cmd
            data.ctrl[4] = 0.0
            
            # Step
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
            
            # Logging everthing so we can later see it
            times.append(t)
            positions.append(q.copy())
            desired_pos.append(q_d.copy())
            errors.append(e.copy())
            vel_cmds.append(vel_cmd.copy())
            
            if i % 1000 == 0:
                print(f"  t={t:5.1f}s | error norm: {np.linalg.norm(e):.6f} rad")
    
    return (np.array(times), np.array(positions), np.array(desired_pos),
            np.array(errors), np.array(vel_cmds))


def plot_pose_regulation(times, positions, errors, vel_cmds, q_target):
    """Plot pose regulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pose Regulation — Velocity Controller", fontsize=14)
    labels = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]
    
    ax = axes[0, 0]
    for j in range(4):
        ax.plot(times, positions[:, j], label=labels[j])
        ax.axhline(y=q_target[j], color=f'C{j}', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Positions"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for j in range(4):
        ax.plot(times, errors[:, j], label=labels[j])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (rad)")
    ax.set_title("Tracking Errors"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for j in range(4):
        ax.plot(times, vel_cmds[:, j], label=labels[j])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Velocity Commands"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(times, np.linalg.norm(errors, axis=1), 'k-', linewidth=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("||e|| (rad)")
    ax.set_title("Norm of Tracking Error"); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pose_regulation_velocity.png", dpi=150)
    print("Saved: pose_regulation_velocity.png")
    plt.show()


def plot_trajectory_tracking(times, positions, desired, errors, vel_cmds):
    """Plot trajectory tracking results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trajectory Tracking — Velocity Controller", fontsize=14)
    labels = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]
    
    ax = axes[0, 0]
    for j in range(4):
        ax.plot(times, positions[:, j], label=f"{labels[j]} actual")
        ax.plot(times, desired[:, j], '--', alpha=0.5, label=f"{labels[j]} desired")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Positions"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for j in range(4):
        ax.plot(times, errors[:, j], label=labels[j])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (rad)")
    ax.set_title("Tracking Errors"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for j in range(4):
        ax.plot(times, vel_cmds[:, j], label=labels[j])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Velocity Commands"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(times, np.linalg.norm(errors, axis=1), 'k-', linewidth=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("||e|| (rad)")
    ax.set_title("Norm of Tracking Error"); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trajectory_tracking_velocity.png", dpi=150)
    print("Saved: trajectory_tracking_velocity.png")
    plt.show()


#Main loop running code
if __name__ == "__main__":
    # Load velocity-controlled model
    model_path = "/home/pinaka/robotis_mujoco_menagerie/robotis_open_manipulator_x/scene_velocity.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"DOF: {model.nv}, Actuators: {model.nu}")
    print(f"Timestep: {model.opt.timestep}s")
    
    # Controller setup
    # Kp = how aggressively position error converts to velocity command
    # Ki = integral gain to eliminate steady-state error
    # vel_limit = max velocity in rad/s (Dynamixel XM430 max is ~4.8 rad/s)
    Kp = np.array([5.0, 5.0, 5.0, 5.0])
    Ki = np.array([0.5, 0.5, 0.5, 0.5])
    
    controller = VelocityController(n_joints=4, Kp=Kp, Ki=Ki, vel_limit=4.8)
    
    #  1: Pose Regulation
    mujoco.mj_resetData(model, data)
    q_target = np.array([0.50, -0.35, 0.20, 0.50])
    
    print("\nCclose for plots.")
    times, pos, err, vcmd = run_pose_regulation(model, data, controller, q_target)
    plot_pose_regulation(times, pos, err, vcmd, q_target)
    
    #2: Trajectory Tracking
    input("\nPress Enter to start trajectory tracking...")
    
    mujoco.mj_resetData(model, data)
    data.qpos[:4] = [0.30, -0.20, 0.25, 0.10]
    mujoco.mj_forward(model, data)
    controller.reset()
    
    times, pos, des, err, vcmd = run_trajectory_tracking(model, data, controller)
    plot_trajectory_tracking(times, pos, des, err, vcmd)
    
    print("\nDone! Check saved plots.")
