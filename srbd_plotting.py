import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

class HumanoidVisualizer:
    def __init__(self, is_static=False):
        """
        Initializes the humanoid visualizer.
        
        Parameters:
          is_static: Boolean flag. If True, uses a fixed coordinate frame.
        """
        self.is_static = is_static
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Global parameters.
        self.body_size = (0.3, 0.5, 1.)  # width, depth, height
        self.foot_length = 0.4
        self.foot_width = 0.2

    @staticmethod
    def euler_rotation_matrix(roll, pitch, yaw):
        """
        Returns the rotation matrix for the given Euler angles.
        Rotation order: roll (x), pitch (y), yaw (z).
        """
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
      
        Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
      
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0, 0, 1]])
      
        return Rz @ Ry @ Rx

    def get_torso_faces(self, pos, size, euler_angles=(0, 0, 0)):
        """
        Returns the Poly3DCollection representing a cuboid for the torso.
        
        Parameters:
          pos: Tuple (x, y, z) for the lower corner of the cuboid.
          size: Tuple (dx, dy, dz) dimensions.
          euler_angles: Tuple (roll, pitch, yaw) in radians.
        """
        dx, dy, dz = size
        x, y, z = pos - np.array([dx/2, dy/2, dz/2])
      
        vertices = np.array([
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz]
        ])
      
        pivot = np.array([x + dx/2.0, y + dy/2.0, z + dz/2.0])
        R = self.euler_rotation_matrix(*euler_angles)
      
        rotated_vertices = []
        for v in vertices:
            rotated = R @ (v - pivot) + pivot
            rotated_vertices.append(rotated)
        rotated_vertices = np.array(rotated_vertices)
      
        faces = [
            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],
            [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],
            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],
            [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],
            [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],
            [rotated_vertices[3], rotated_vertices[0], rotated_vertices[4], rotated_vertices[7]]
        ]
      
        cuboid = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.7)
        return cuboid

    def foot_vertices(self, center, angle):
        """
        Returns 3D vertices for a foot polygon given its center, dimensions, and rotation.
        
        Parameters:
          center: Tuple or array (x, y) representing the foot center.
          angle: Rotation angle in radians for the foot.
        """
        cx, cy = center
        hl = self.foot_length / 2.0
        hw = self.foot_width / 2.0
        local_corners = np.array([
            [-hl, -hw],
            [ hl, -hw],
            [ hl,  hw],
            [-hl,  hw]
        ])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        world_corners = np.dot(local_corners, R.T) + np.array([cx, cy])
        return [[pt[0], pt[1], 0] for pt in world_corners]

    def set_left_foot_pose(self, x, y, angle):
        """
        Returns the vertices of the left foot based on the given center.
        """
        center = np.array([x, y])
        return self.foot_vertices(center, angle)

    def set_right_foot_pose(self, x, y, angle):
        """
        Returns the vertices of the right foot based on the given center.
        """
        center = np.array([x, y])
        return self.foot_vertices(center, angle)

    def draw_humanoid(self, torso_pos, torso_euler, left_foot_center, left_foot_angle,
                      right_foot_center, right_foot_angle):
        """
        Updates the plot with the provided torso and feet poses.
        
        Parameters:
          torso_pos: Tuple (x, y, z) position of the torso (lower corner).
          torso_euler: Tuple (roll, pitch, yaw) in radians for the torso.
          left_foot_center: Tuple (x, y) for the left foot center.
          left_foot_angle: Rotation angle in radians for the left foot.
          right_foot_center: Tuple (x, y) for the right foot center.
          right_foot_angle: Rotation angle in radians for the right foot.
        """
        self.ax.clear()
        # Set axis limits based on the static flag.
        if self.is_static:
            margin = 3
            self.ax.set_xlim(-margin, margin)
            self.ax.set_ylim(-margin, margin)
        else:
            x, y, _ = torso_pos
            margin = 3
            self.ax.set_xlim(x - margin, x + margin)
            self.ax.set_ylim(y - margin, y + margin)
            
        self.ax.set_zlim(0, 3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Humanoid with Two Feet')
        
        # Draw torso.
        cuboid = self.get_torso_faces(torso_pos, self.body_size, torso_euler)
        self.ax.add_collection3d(cuboid)
        
        # Draw feet.
        left_foot = self.set_left_foot_pose(left_foot_center[0], left_foot_center[1], left_foot_angle)
        right_foot = self.set_right_foot_pose(right_foot_center[0], right_foot_center[1], right_foot_angle)
        
        left_foot_poly = Poly3DCollection([left_foot], facecolors='brown', linewidths=1, edgecolors='k', alpha=0.8)
        right_foot_poly = Poly3DCollection([right_foot], facecolors='brown', linewidths=1, edgecolors='k', alpha=0.8)
        
        self.ax.add_collection3d(left_foot_poly)
        self.ax.add_collection3d(right_foot_poly)
        
        plt.draw()
        plt.pause(0.0001)

    def update_and_plot_humanoid(self, pose):
        """
        Accepts a pose dictionary and updates the visualization.
        
        Pose keys:
          'torso_pos', 'torso_euler',
          'left_foot_center', 'left_foot_angle',
          'right_foot_center', 'right_foot_angle'
        """
        self.draw_humanoid(pose['torso_pos'], pose['torso_euler'],
                           pose['left_foot_center'], pose['left_foot_angle'],
                           pose['right_foot_center'], pose['right_foot_angle'])
        
# # Example usage (if running this file directly):
# if __name__ == '__main__':
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Create an instance of the visualizer.
#     visualizer = HumanoidVisualizer(is_static=False)

#     def get_poses(t):
#         return {
#             'torso_pos': (t, t, 1 + 0. * abs(np.sin(t))),
#             'torso_euler': (t * 0.1, t * 0.1, t * 0.1),
#             'left_foot_center': (t + 0.4, t - 0.15),
#             'left_foot_angle': 0,
#             'right_foot_center': (t, t + 0.15),
#             'right_foot_angle': 0
#         }

#     plt.ion()
#     for t in np.linspace(0, 10, 100):
#         visualizer.update_and_plot_humanoid(get_poses(t))
#     plt.ioff()
#     plt.show()
