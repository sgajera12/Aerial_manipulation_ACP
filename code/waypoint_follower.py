import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class VelocityController:
    def __init__(self, n_joints, Kp, Ki=None, vel_limit=3):
        self.n_joints = n_joints
        self.Kp = np.atleast_1d(Kp) * np.ones(n_joints)
        self.Ki = np.atleast_1d(Ki) * np.ones(n_joints) if Ki is not None else None
        self.vel_limit = vel_limit
        self.error_integral = np.zeros(n_joints)

    def reset(self):
        self.error_integral = np.zeros(self.n_joints)

    def compute_velocity(self, q, q_d, qd_d=None, dt=0.002):
        e = q_d - q
        vel_cmd = self.Kp * e
        if qd_d is not None:
            vel_cmd += qd_d
        if self.Ki is not None:
            self.error_integral += e * dt
            self.error_integral = np.clip(self.error_integral, -1.0, 1.0)
            vel_cmd += self.Ki * self.error_integral
        vel_cmd = np.clip(vel_cmd, -self.vel_limit, self.vel_limit)
        return vel_cmd, e


# WAYPOINT DEFINITIONS
# Each waypoint: [Joint1, Joint2, Joint3, Joint4] in radians
waypoints = [
    {"name": "Home","q": [0.0,0.0, 0.0,0.0 ]},
    {"name": "Look Left","q": [1.0,-0.3,0.2,0.5 ]},
    {"name": "Reach Forward","q": [0.0,-0.5, 0.3,0.8 ]},
    {"name": "Look Right","q": [-1.0,-0.3, 0.2, 0.5 ]},
    {"name": "Reach Up", "q": [0.0,-1.0, 0.5,0.3 ]},
    {"name": "Pick Position", "q": [0.5,0.2, -0.3,1.2 ]},
    {"name": "Home","q": [0.0,0.0,0.0,0.0 ]},
]

# How close the robot must be to a waypoint before moving to next (radians)
ARRIVAL_THRESHOLD = 0.01
# Max time to spend at each waypoint before moving on (seconds)
MAX_WAIT_TIME = 4.0
# Time to pause at waypoint after arriving (seconds)
SETTLE_TIME = 1.5


# MAIN
if __name__ == "__main__":
    model_path = "/home/pinaka/robotis_mujoco_menagerie/robotis_open_manipulator_x/scene_velocity.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # Controller
    Kp = np.array([2.5, 2.5, 2.5, 2.5])
    Ki = np.array([0.0, 0.0, 0.0, 0.0])
    controller = VelocityController(n_joints=4, Kp=Kp, Ki=Ki)

    # Logging
    times, positions, errors, vel_cmds = [], [], [], []
    waypoint_times = []  # when each waypoint was reached
    waypoint_labels = []

    # State machine
    current_wp = 0
    wp_arrived = False
    wp_arrive_time = 0.0

    print("WAYPOINT FOLLOWING")
    for i, wp in enumerate(waypoints):
        print(f"  WP {i}: {wp['name']:20s} q = {wp['q']}")
    print(f"\nArrival threshold: {ARRIVAL_THRESHOLD} rad")
    print(f"Settle time at waypoint: {SETTLE_TIME} s")
    print(f"\nMoving to WP 0: {waypoints[0]['name']}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and current_wp < len(waypoints):
            t = data.time
            q = data.qpos[:4]
            q_target = np.array(waypoints[current_wp]["q"])

            # Check if arrived at current waypoint
            error_norm = np.linalg.norm(q_target - q)

            if not wp_arrived and error_norm < ARRIVAL_THRESHOLD:
                wp_arrived = True
                wp_arrive_time = t
                waypoint_times.append(t)
                waypoint_labels.append(waypoints[current_wp]["name"])
                print(f" Arrived at WP {current_wp}: {waypoints[current_wp]['name']} "
                      f"(t={t:.2f}s, error={error_norm:.4f} rad)")

            # If arrived and settled, move to next waypoint
            if wp_arrived and (t - wp_arrive_time) > SETTLE_TIME:
                current_wp += 1
                wp_arrived = False
                controller.reset()  # reset integral for new target
                if current_wp < len(waypoints):
                    print(f"\nMoving to WP {current_wp}: {waypoints[current_wp]['name']}")

            # Timeout: move to next even if not arrived
            if not wp_arrived and current_wp < len(waypoints):
                time_at_wp = t - (waypoint_times[-1] + SETTLE_TIME if waypoint_times else 0)
                if time_at_wp > MAX_WAIT_TIME:
                    print(f" Timeout at WP {current_wp}, moving on (error={error_norm:.4f})")
                    current_wp += 1
                    wp_arrived = False
                    controller.reset()
                    if current_wp < len(waypoints):
                        print(f"\nMoving to WP {current_wp}: {waypoints[current_wp]['name']}")

            # Compute and apply velocity command
            if current_wp < len(waypoints):
                q_target = np.array(waypoints[current_wp]["q"])
                vel_cmd, e = controller.compute_velocity(q, q_target, dt=dt)
                data.ctrl[:4] = vel_cmd
            else:
                data.ctrl[:4] = 0.0  # stop

            data.ctrl[4] = 0.0  # gripper

            # Step
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

            # Log
            times.append(t)
            positions.append(q.copy())
            errors.append((np.array(waypoints[min(current_wp, len(waypoints)-1)]["q"]) - q).copy())
            vel_cmds.append(data.ctrl[:4].copy())

    print("\n All waypoints visited")

    # PLOT
    times = np.array(times)
    positions = np.array(positions)
    errors = np.array(errors)
    vel_cmds = np.array(vel_cmds)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Waypoint Following — Velocity Controller", fontsize=14)
    labels = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]

    # Joint positions with waypoint markers
    ax = axes[0]
    for j in range(4):
        ax.plot(times, positions[:, j], label=labels[j])
    # Draw vertical lines at waypoint arrivals
    for wt, wl in zip(waypoint_times, waypoint_labels):
        ax.axvline(x=wt, color='gray', linestyle='--', alpha=0.5)
        ax.text(wt + 0.1, ax.get_ylim()[1] * 0.9, wl, fontsize=7, rotation=45)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Positions"); ax.legend(); ax.grid(True, alpha=0.3)

    # Error norm
    ax = axes[1]
    ax.plot(times, np.linalg.norm(errors, axis=1), 'k-', linewidth=1.5)
    for wt in waypoint_times:
        ax.axvline(x=wt, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=ARRIVAL_THRESHOLD, color='r', linestyle=':', label=f'Threshold ({ARRIVAL_THRESHOLD} rad)')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("||e|| (rad)")
    ax.set_title("Tracking Error Norm"); ax.legend(); ax.grid(True, alpha=0.3)

    # Velocity commands
    ax = axes[2]
    for j in range(4):
        ax.plot(times, vel_cmds[:, j], label=labels[j])
    for wt in waypoint_times:
        ax.axvline(x=wt, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Velocity Commands"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("waypoint_following.png", dpi=150)
    print("Saved: waypoint_following.png")
    plt.show()
