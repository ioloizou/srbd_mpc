import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from srbd_plotting import SRBDVisualizer
from gait_planner import GaitPlanner

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

def main():
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

if __name__ == '__main__':
    main()
