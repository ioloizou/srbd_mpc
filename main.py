import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from srbd_plotting import SRBDVisualizer
from gait_planner import GaitPlanner
import mpc

def get_poses(t, gait_phases):
    # Find the current gait phase based on time.
    phase = None
    for ph in gait_phases:
        if ph["start_time"] <= t < ph["end_time"]:
            phase = ph
            break
    if phase is None:
        phase = gait_phases[-1]

    # Torso pose and grf.
    pose = {
        'torso_pos': (t * 0.25, 0.05 * np.sin(4*t), 1. + 0.1 * np.sin(t)),
        'torso_euler': (0.1 * np.sin(t), 0.1 * np.sin(t), 0.1 * np.sin(t)),
        'left_grf': (1 * np.sin(t), 1 * abs(np.sin(t)), 30 * abs(np.sin(t))),
        'right_grf': (2 * np.sin(t), 3 * abs(np.sin(t)), 20 * abs(np.sin(t)))
    }

    # Left foot in stance.
    if phase["support_leg"] in ["both", "left"]:
        pose["left_foot_center"] = phase["left_foot"]
        pose["left_foot_angle"] = 0
        if "adapted_left_footstep" not in phase:
            adapted_offset = np.random.rand(2) * 0.1
            adapted_left = phase["left_foot"] + adapted_offset
            phase["adapted_left_footstep"] = [{"foot_center": adapted_left.tolist(), "foot_angle": 0}]
        pose["adapted_left_footstep"] = phase["adapted_left_footstep"]

    # Right foot in stance.
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
    
    # Create a time array from 0 to 8.5 s for 60 frames per second.
    t_frames = np.linspace(0, 8.5, 200)

    # Animation function: update both figures.
    def animate(t):
        pose = get_poses(t, gait_phases)
        visualizer.update_and_plot_humanoid(pose)
        # visualizer.update_top_view(gait_phases, pose['torso_pos'])
        # Return an empty list (or any updated artists if needed).
        return []

    # # Animate the 2D figure at 60 fps.
    # ani = animation.FuncAnimation(visualizer.fig3d, animate, frames=t_frames, 
    #                               interval=5, blit=False, repeat=False)


    # Animate the 2D figure at 60 fps.
    ani = animation.FuncAnimation(visualizer.fig2d, animate, frames=t_frames, 
                                  interval=100, blit=False, repeat=False)

    # Draw the figures.
    plt.show()

def main():
    
    # plot()

    SRBD_mpc = mpc.MPC()
    planner = GaitPlanner(0.5, 0.8, 0.3, 0.2, 10)
    gait_phases = planner.plan_gait()
    # print("Gait phases shape:", np.array(gait_phases).shape)
    # print("Gait phases:", gait_phases)
    
    # Create the reference trajectory for MPC.
    dt = SRBD_mpc.dt
    horizon = SRBD_mpc.HORIZON_LENGTH
    state_dim = SRBD_mpc.NUM_STATES    
    
    # Initialize x_ref as zeros.
    x_ref = np.zeros((horizon, state_dim))
    
    # Construct the trajectory:
    # - Position starts at 0,0,0. The x position increases linearly with the commanded vx=1 m/s.
    # - Orientation remains 0,0,0.
    # - All velocities are zero initially except for the x velocity, which is 1 m/s.
    for i in range(horizon):
        pos = i * dt
        x_ref[i, 3] = pos         # x position: 1 m/s * t
        x_ref[i, 5] = 1         # vx: constant 1 m/s

    # Build the c_horizon vector for the current MPC horizon based on time progression.
    dt = SRBD_mpc.dt
    horizon = SRBD_mpc.HORIZON_LENGTH
    current_time = 0.0  # This should be updated each MPC solve if needed.

    c_horizon = []
    contact_horizon = []
    for i in range(horizon):
        t_i = current_time + i * dt
        # Find the gait phase corresponding to t_i.
        phase = None
        for ph in gait_phases:
            if ph["start_time"] <= t_i < ph["end_time"]:
                phase = ph
                break
        if phase is None:
            phase = gait_phases[-1]

        left = phase["left_foot"]
        right = phase["right_foot"]
        foot_vec = np.concatenate((left, right), axis=0)
        c_horizon.append(foot_vec)

        # Define contact: 1 if foot is in stance, 0 if in swing.
        left_contact = 1 if phase["support_leg"] in ["both", "left"] else 0
        right_contact = 1 if phase["support_leg"] in ["both", "right"] else 0
        contact_horizon.append([left_contact, right_contact])
        
    c_horizon = np.array(c_horizon)
    print("c_horizon:", c_horizon)
    
    contact_horizon = np.array(contact_horizon)
    print("contact_horizon:", contact_horizon)

    p_com_horizon = x_ref[:, 3:6]
    # print("p_com_horizon:", p_com_horizon)
    
    SRBD_mpc.init_matrices()
    
    SRBD_mpc.x0 = x_ref[0]
    
    SRBD_mpc.extract_psi(x_ref)
    print("psi in degrees:", np.degrees(SRBD_mpc.psi))
    SRBD_mpc.rotation_matrix_T()
    print("Rotation Matrix: \n", SRBD_mpc.rotation_z)
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

    SRBD_mpc.compute_rollout()







if __name__ == '__main__':
    main()
