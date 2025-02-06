import srbd_plotting as srbd_plt
import matplotlib.pyplot as plt
import numpy as np

def get_poses(t):
        return {
            'torso_pos': (t, t, 1 + 0. * abs(np.sin(t))),
            'torso_euler': (t * 0.1, t * 0.1, t * 0.1),
            'left_foot_center': (t + 0.4, t - 0.15),
            'left_foot_angle': 0,
            'right_foot_center': (t, t + 0.15),
            'right_foot_angle': 0
        }

def main():    
    is_interactive = True
    
    if is_interactive:
        plt.ion()
        for t in np.linspace(0, 10, 100):
            srbd_plt.update_and_plot_humanoid(get_poses(t))
    else:
        srbd_plt.update_and_plot_humanoid(get_poses(1))
        plt.show()

if __name__ == '__main__':
    main()