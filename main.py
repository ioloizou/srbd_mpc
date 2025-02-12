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



def dummy_standing_quad(SRBD_mpc):
    # Define constant foot positions for the whole horizon.
    # 1st foot (front left):  0.2,  0.2
    # 2nd foot (front right): 0.2, -0.2
    # 3rd foot (rear left):  -0.2,  0.2
    # 4th foot (rear right): -0.2, -0.2
    # Here, we pack the positions as a single row:
    foot_positions = np.array([0.2, 0.2, 0., 0.2, -0.2, 0., -0.2, 0.2, 0., -0.2, -0.2, 0.])*4.
    
    # For each time step in the horizon, use the same foot positions.
    c_horizon = np.tile(foot_positions, (SRBD_mpc.HORIZON_LENGTH, 1))
    
    # All contacts are active (1 indicates foot in stance).
    # Four foot contacts for each time step.
    contact_horizon = np.ones((SRBD_mpc.HORIZON_LENGTH, 4))
    
    # For debugging or further processing.
    # print("Dummy standing quad - c_horizon:")
    # print(c_horizon)
    # print("Dummy standing quad - contact_horizon:")
    # print(contact_horizon)

    return c_horizon, contact_horizon

def main():
    
    # plot()

    SRBD_mpc = mpc.MPC()
    # planner = GaitPlanner(0.5, 0.8, 0.3, 0.2, 10)
    # planner = GaitPlanner(0.1, 0.1, 0.3, 0.2, 10)
    # gait_phases = planner.plan_gait()
    
    SRBD_mpc.init_matrices()
   
    # Initialize SRBD_mpc.x_ref_hor as zeros.
    SRBD_mpc.x_ref_hor = np.zeros((SRBD_mpc.HORIZON_LENGTH, SRBD_mpc.NUM_STATES))

    SRBD_mpc.x_ref_hor[:, -1] = SRBD_mpc.g
    SRBD_mpc.x_ref_hor[:, 5] = 1.     # z position: constant 1 m
    # SRBD_mpc.x_ref_hor[:, 6] = 1      # x velocity: constant 1 m/s
    

    
    current_time = 0.0
    total_duration = 20.
    # total_duration = gait_phases[-1]["end_time"]

    com_hist = []

    while current_time < total_duration:
        print("Index:", int(current_time / SRBD_mpc.dt))
        dt = SRBD_mpc.dt  # MPC time step
        

        if current_time == 0.0:
            # I should be carefull here i was giving initial velocity as 1 m/s
            SRBD_mpc.x0 = SRBD_mpc.x_ref_hor[0].copy()
        else:
            SRBD_mpc.x0 = SRBD_mpc.x_opt[-1].copy()
            print("x0:", SRBD_mpc.x0)

        # Construct the trajectory relative to the current time:
        # - x position increases linearly based on current_time
        # for i in range(SRBD_mpc.HORIZON_LENGTH):
        #     t_i = current_time + i * dt
        #     SRBD_mpc.x_ref_hor[i, 3] = t_i    # x position: 1 m/s * (current time + t_offset)
             
        
        # c_horizon = []
        # contact_horizon = []
        # for i in range(SRBD_mpc.HORIZON_LENGTH):
        #     t_i = current_time + i * dt
        #     # Find the gait phase corresponding to t_i.
        #     phase = None
        #     for ph in gait_phases:
        #         if ph["start_time"] <= t_i < ph["end_time"]:
        #             phase = ph
        #             break
        #     if phase is None:
        #         phase = gait_phases[-1]

        #     left = phase["left_foot"]
        #     right = phase["right_foot"]
        #     foot_vec = np.concatenate((left, right), axis=0)
        #     c_horizon.append(foot_vec)

        #     # Define contact: 1 if foot is in stance, 0 if in swing.
        #     left_contact = 1 if phase["support_leg"] in ["both", "left"] else 0
        #     right_contact = 1 if phase["support_leg"] in ["both", "right"] else 0
        #     contact_horizon.append([left_contact, right_contact])
            
        # c_horizon = np.array(c_horizon)
        # # print("c_horizon:", c_horizon)
        
        # contact_horizon = np.array(contact_horizon)
        # # print("contact_horizon:", contact_horizon)

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
        
        # Print current COM position and footstep locations
        print("COM:", SRBD_mpc.x_opt[0, 3:6])
        
        # Print the current left and right footstep locations from c_horizon
        # print("Left foot:", c_horizon[0, :2])
        # print("Right foot:", c_horizon[0, 2:])

        
        # Store the first state of the rollout (i.e. the current COM)
        com_hist.append(SRBD_mpc.x_opt[0, 3:6].copy())
        # com_hist.append(SRBD_mpc.x_ref_hor[0, 3:6].copy())
        
        current_time += dt

        # print("Current time:", current_time)


    # After simulation finishes, plot the COM trajectory in 3D.
    com_hist = np.array(com_hist)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(com_hist[:,0], com_hist[:,1], com_hist[:,2], marker='o')
    ax.set_title("Center of Mass Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == '__main__':
    main()
