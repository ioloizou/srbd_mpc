from test import update_torso_pose, update_left_foot_pose, update_right_foot_pose
import time

def main():
    # For example, update the humanoid's pose over time.
    for step in range(100):
        # Compute new torso and foot positions.
        x = step * 0.05
        y = 0
        z = 1
        
        # Update the torso pose.
        update_torso_pose((x, y, z), (0, 0, 0))

        # Update the left foot pose.
        update_left_foot_pose((x, y), 0)

        # Update the right foot pose.
        update_right_foot_pose((x, y), 0)
        
        time.sleep(0.05)

if __name__ == '__main__':
    main()