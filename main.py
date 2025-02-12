import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from srbd_plotting import SRBDVisualizer
from gait_planner import GaitPlanner
import mpc

def get_poses(t, gait_phases):
    # (existing implementation)
    phase = None
    for ph in gait_phases:
        if ph["start_time"] <= t < ph["end_time"]:
            phase = ph
            break
    if phase is None:
        phase = gait_phases[-1]

    pose = {
        'torso_pos': (t * 0.25, 0.05 * np.sin(4*t), 1. + 0.1 * np.sin(t)),
        'torso_euler': (0.1 * np.sin(t), 0.1 * np.sin(t), 0.1 * np.sin(t)),
        'left_grf': (1 * np.sin(t), 1 * abs(np.sin(t)), 30 * abs(np.sin(t))),
        'right_grf': (2 * np.sin(t), 3 * abs(np.sin(t)), 20 * abs(np.sin(t)))
    }
    if phase["support_leg"] in ["both", "left"]:
        pose["left_foot_center"] = phase["left_foot"]
        pose["left_foot_angle"] = 0
        if "adapted_left_footstep" not in phase:
            adapted_offset = np.random.rand(2) * 0.1
            adapted_left = phase["left_foot"] + adapted_offset
            phase["adapted_left_footstep"] = [{"foot_center": adapted_left.tolist(), "foot_angle": 0}]
        pose["adapted_left_footstep"] = phase["adapted_left_footstep"]

    if phase["support_leg"] in ["both", "right"]:
        pose["right_foot_center"] = phase["right_foot"]
        pose["right_foot_angle"] = 0
        if "adapted_right_footstep" not in phase:
            adapted_offset = np.random.rand(2) * 0.1
            adapted_right = phase["right_foot"] + adapted_offset
            phase["adapted_right_footstep"] = [{"foot_center": adapted_right.tolist(), "foot_angle": 0}]
        pose["adapted_right_footstep"] = phase["adapted_right_footstep"]

    print("Time: ", t)
    if "left_foot_center" in pose:
        print("Left foot (stance): ", pose["left_foot_center"])
    else:
        print("Left foot: Swing")
    if "right_foot_center" in pose:
        print("Right foot (stance): ", pose["right_foot_center"])
    else:
        print("Right foot: Swing")
    return pose

def plot():
    # Initialize the gait planner.
    planner = GaitPlanner(0.5, 0.8, 0.3, 0.2, 10)
    gait_phases = planner.plan_gait()

    visualizer = SRBDVisualizer(is_static=False)
    
    # Create a time array from 0 to 8.5 s for animation.
    t_frames = np.linspace(0, 8.5, 200)

    def animate(t):
        pose = get_poses(t, gait_phases)
        visualizer.update_and_plot_humanoid(pose)
        return []

    ani = animation.FuncAnimation(visualizer.fig2d, animate, frames=t_frames, 
                                  interval=100, blit=False, repeat=False)
    plt.show()

def dummy_standing_quad(SRBD_mpc):
    # Dummy foot positions.
    foot_positions = np.array([0.2, 0.2, 0., 0.2, -0.2, 0., -0.2, 0.2, 0., -0.2, -0.2, 0.])*4.
    c_horizon = np.tile(foot_positions, (SRBD_mpc.HORIZON_LENGTH, 1))
    contact_horizon = np.ones((SRBD_mpc.HORIZON_LENGTH, 4))
    return c_horizon, contact_horizon

def plot_all_trajectories(com_hist, orient_hist, vel_hist, orient_deriv_hist, dt):
    n_points = com_hist.shape[0]
    time_array = np.arange(n_points) * dt

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Center of Mass Trajectory
    ax = axs[0, 0]
    ax.plot(time_array, com_hist[:, 0], marker='o', label="X")
    ax.plot(time_array, com_hist[:, 1], marker='o', label="Y")
    ax.plot(time_array, com_hist[:, 2], marker='o', label="Z")
    ax.set_title("Center of Mass Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.grid(True)
    ax.legend()

    # Orientation Trajectory
    ax = axs[0, 1]
    ax.plot(time_array, orient_hist[:, 0], marker='o', label="Roll")
    ax.plot(time_array, orient_hist[:, 1], marker='o', label="Pitch")
    ax.plot(time_array, orient_hist[:, 2], marker='o', label="Yaw")
    ax.set_title("COM Orientation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
    ax.legend()

    # COM Velocity Trajectory
    ax = axs[1, 0]
    ax.plot(time_array, vel_hist[:, 0], marker='o', label="Vx")
    ax.plot(time_array, vel_hist[:, 1], marker='o', label="Vy")
    ax.plot(time_array, vel_hist[:, 2], marker='o', label="Vz")
    ax.set_title("COM Velocity Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.grid(True)
    ax.legend()

    # Orientation Derivative Trajectory
    ax = axs[1, 1]
    ax.plot(time_array, orient_deriv_hist[:, 0], marker='o', label="Roll Rate")
    ax.plot(time_array, orient_deriv_hist[:, 1], marker='o', label="Pitch Rate")
    ax.plot(time_array, orient_deriv_hist[:, 2], marker='o', label="Yaw Rate")
    ax.set_title("Orientation Derivative Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (rad/s)")
    ax.grid(True)
    ax.legend()

    plt.show()
    
def plot_ground_reaction_forces(grf_hist, dt):
    grf_hist = np.array(grf_hist)
    time_array = np.arange(grf_hist.shape[0]) * dt
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(time_array, grf_hist[:, 0], label='Fx (Left)')
    axs[0].plot(time_array, grf_hist[:, 1], label='Fy (Left)')
    axs[0].plot(time_array, grf_hist[:, 2], label='Fz (Left)')
    axs[0].set_title('Left Leg Ground Reaction Forces')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Force (N)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time_array, grf_hist[:, 3], label='Fx (Right)')
    axs[1].plot(time_array, grf_hist[:, 4], label='Fy (Right)')
    axs[1].plot(time_array, grf_hist[:, 5], label='Fz (Right)')
    axs[1].set_title('Right Leg Ground Reaction Forces')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Force (N)')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    SRBD_mpc = mpc.MPC()
    SRBD_mpc.init_matrices()

    # Setup reference horizon (dummy initialization).
    SRBD_mpc.x_ref_hor = np.zeros((SRBD_mpc.HORIZON_LENGTH, SRBD_mpc.NUM_STATES))
    SRBD_mpc.x_ref_hor[:, -1] = SRBD_mpc.g
    SRBD_mpc.x_ref_hor[:, 5] = 1.     # For example: constant 1 m z position

    current_time = 0.0
    total_duration = 2.
    dt = SRBD_mpc.dt

    com_hist = []
    orient_hist = []
    vel_hist = []
    orient_deriv_hist = []
    grf_hist = []


    while current_time < total_duration:
        print("Index:", int(current_time / dt))
        if current_time == 0.0:
            SRBD_mpc.x0 = SRBD_mpc.x_ref_hor[0].copy()
        else:
            SRBD_mpc.x0 = SRBD_mpc.x_opt[-1].copy()
            print("x0:", SRBD_mpc.x0)

        p_com_horizon = SRBD_mpc.x_ref_hor[:, 3:6].copy()
        c_horizon, contact_horizon = dummy_standing_quad(SRBD_mpc)

        SRBD_mpc.extract_psi()
        SRBD_mpc.rotation_matrix_T()
        SRBD_mpc.set_Q()
        SRBD_mpc.set_R()
        SRBD_mpc.calculate_A_continuous()
        SRBD_mpc.calculate_A_discrete()
        SRBD_mpc.calculate_B_continuous(c_horizon, p_com_horizon)
        SRBD_mpc.calculate_B_discrete()
        SRBD_mpc.calculate_Aqp()
        SRBD_mpc.calculate_Bqp()
        SRBD_mpc.calculate_Ac()
        SRBD_mpc.calculate_bounds(contact_horizon)
        SRBD_mpc.calculate_hessian()
        SRBD_mpc.calculate_gradient()
        SRBD_mpc.solve_qp()

        if current_time == 0.0:
            SRBD_mpc.x_opt[0, :] = np.squeeze(SRBD_mpc.x0.copy())
        else:
            SRBD_mpc.x_opt[0, :] = np.squeeze(SRBD_mpc.x_opt[-1, :].copy())
        SRBD_mpc.compute_rollout()

        # Assume the COM position is in columns 3:6.
        com = SRBD_mpc.x_opt[0, 3:6].copy()
        com_hist.append(com)

        # For demonstration, assume orientation (roll, pitch, yaw) is computed or available.
        # Here we generate dummy data.
        orientation = SRBD_mpc.x_opt[0, 0:3].copy()
        orient_hist.append(orientation)

        velocity = SRBD_mpc.x_opt[0, 6:9].copy()
        vel_hist.append(velocity)
        

        orient_deriv = SRBD_mpc.x_opt[0, 9:12].copy()
        orient_deriv_hist.append(orient_deriv)
        
        grf_hist.append(SRBD_mpc.u_opt[0].copy())


        print("COM:", com)
        current_time += dt

    com_hist = np.array(com_hist)
    orient_hist = np.array(orient_hist)
    vel_hist = np.array(vel_hist)
    orient_deriv_hist = np.array(orient_deriv_hist)
    grf_hist = np.array(grf_hist)


    plot_all_trajectories(com_hist, orient_hist, vel_hist, orient_deriv_hist, dt)
    plot_ground_reaction_forces(grf_hist, dt)
if __name__ == '__main__':
    main()
