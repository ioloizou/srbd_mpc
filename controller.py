import srbd_plotting as srbd_plt
import matplotlib.pyplot as plt
import numpy as np

def main():    
    # Turn on interactive mode.
    # plt.ion() 

    # For example, update the humanoid's pose over time.
    # for t in np.linspace(0, 10, 100):  
        # Compute new torso and foot poses.

    t=1
    poses = {
    'torso_pos': (t, t, 1 + 0.5 * abs(np.sin(t))),
    'torso_euler': (t * 0.1, t * 0.1, t * 0.1),
    'left_foot_center': (t, t),
    'left_foot_angle': 0,
    'right_foot_center': (t  + 0.2, t + 0.4),
    'right_foot_angle': 0
    }

    # Update the plot with the new pose.
    srbd_plt.update_and_plot_humanoid(poses)

    # Keep the plot open until the user closes it.
    plt.show()

if __name__ == '__main__':
    main()