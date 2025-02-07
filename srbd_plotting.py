import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

class SRBDVisualizer:
  def __init__(self, is_static=False):
    """
    Initializes the SRBD humanoid visualizer.
    
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
    Returns 3D vertices for a foot polygon given its center and rotation.
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

  def draw_torso(self, torso_pos, torso_euler):
    """
    Draws the torso of the humanoid.
    """
    cuboid = self.get_torso_faces(np.array(torso_pos), self.body_size, torso_euler)
    self.ax.add_collection3d(cuboid)

  def draw_left_foot(self, left_foot_center, left_foot_angle):
    """
    Draws the left foot of the humanoid.
    """
    left_foot = self.set_left_foot_pose(left_foot_center[0], left_foot_center[1], left_foot_angle)
    left_foot_poly = Poly3DCollection([left_foot], facecolors='brown',
                      linewidths=1, edgecolors='k', alpha=0.8)
    self.ax.add_collection3d(left_foot_poly)

  def draw_right_foot(self, right_foot_center, right_foot_angle):
    """
    Draws the right foot of the humanoid.
    """
    right_foot = self.set_right_foot_pose(right_foot_center[0], right_foot_center[1], right_foot_angle)
    right_foot_poly = Poly3DCollection([right_foot], facecolors='brown',
                       linewidths=1, edgecolors='k', alpha=0.8)
    self.ax.add_collection3d(right_foot_poly)

  def draw_ground_reaction_force(self, pos, force, color='black'):
    """
    Draws an arrow representing the ground reaction force.
    """
    self.ax.quiver(pos[0], pos[1], pos[2],
             force[0], force[1], force[2],
             color=color, length=0.05)

  def draw_left_grf(self, left_foot_center, left_grf):
    """
    Draws the left foot ground reaction force.
    """
    pos = (left_foot_center[0], left_foot_center[1], 0)
    self.draw_ground_reaction_force(pos, left_grf, color='red')

  def draw_right_grf(self, right_foot_center, right_grf):
    """
    Draws the right foot ground reaction force.
    """
    pos = (right_foot_center[0], right_foot_center[1], 0)
    self.draw_ground_reaction_force(pos, right_grf, color='blue')

  def draw_adapted_left_footstep(self, adapted_left_steps):
    """
    Draws the adapted left footsteps in green.
    
    Parameters:
      adapted_left_steps: List of dictionaries where each contains:
        'foot_center': Tuple (x, y) for the foot center.
        'foot_angle': Rotation angle in radians.
    """
    for step in adapted_left_steps:
      foot_center = step['foot_center']
      foot_angle = step['foot_angle']
      foot = self.foot_vertices(foot_center, foot_angle)
      adapted_left_poly = Poly3DCollection([foot], facecolors='green',
                              linewidths=1, edgecolors='k', alpha=0.8)
      self.ax.add_collection3d(adapted_left_poly)

  def draw_adapted_right_footstep(self, adapted_right_steps):
    """
    Draws the adapted right footsteps in magenta.
    
    Parameters:
      adapted_right_steps: List of dictionaries where each contains:
        'foot_center': Tuple (x, y) for the foot center.
        'foot_angle': Rotation angle in radians.
    """
    for step in adapted_right_steps:
      foot_center = step['foot_center']
      foot_angle = step['foot_angle']
      foot = self.foot_vertices(foot_center, foot_angle)
      adapted_right_poly = Poly3DCollection([foot], facecolors='magenta',
                               linewidths=1, edgecolors='k', alpha=0.8)
      self.ax.add_collection3d(adapted_right_poly)

  def update_and_plot_humanoid(self, pose):
    """
    Updates the plot with separate calls for torso, left/right feet,
    ground reaction forces, and adapted footsteps.
    
    Pose keys:
      'torso_pos', 'torso_euler',
      'left_foot_center', 'left_foot_angle',
      'right_foot_center', 'right_foot_angle',
      'left_grf', 'right_grf' (optional),
      'adapted_left_footstep' (optional): List of adapted left footsteps.
      'adapted_right_footstep' (optional): List of adapted right footsteps.
    """
    self.ax.clear()
    # Set axis limits.
    if self.is_static:
      margin = 3
      self.ax.set_xlim(-margin, margin)
      self.ax.set_ylim(-margin, margin)
    else:
      x, y, _ = pose['torso_pos']
      margin = 3
      self.ax.set_xlim(x - margin, x + margin)
      self.ax.set_ylim(y - margin, y + margin)
    self.ax.set_zlim(0, 3)
    self.ax.set_xlabel('X')
    self.ax.set_ylabel('Y')
    self.ax.set_zlabel('Z')
    self.ax.set_title('3D Humanoid SRBD Visualization')
    
    # Draw torso.
    self.draw_torso(pose['torso_pos'], pose['torso_euler'])
    
    # Draw feet.
    if 'left_foot_center' in pose:
      self.draw_left_foot(pose['left_foot_center'], pose['left_foot_angle'])
    if 'right_foot_center' in pose:
      self.draw_right_foot(pose['right_foot_center'], pose['right_foot_angle'])
    
    # Draw ground reaction forces.
    if 'left_grf' in pose and 'left_foot_center' in pose:
      self.draw_left_grf(pose['left_foot_center'], pose['left_grf'])
    if 'right_grf' in pose and 'right_foot_center' in pose:
      self.draw_right_grf(pose['right_foot_center'], pose['right_grf'])
    
    # Draw adapted footsteps if provided.
    if 'adapted_left_footstep' in pose:
      self.draw_adapted_left_footstep(pose['adapted_left_footstep'])
    if 'adapted_right_footstep' in pose:
      self.draw_adapted_right_footstep(pose['adapted_right_footstep'])
    
    plt.draw()
    plt.pause(0.0001)

  def plot_top_view(self, gait_phases, torso_pos):
    """
    Updates (or creates once) a 2D top view showing:
      - Planned footsteps as rectangles (left: blue, right: red),
      - Adapted footsteps as rectangles (both in green),
      - Torso pivot point (COM) in magenta, and
      - A line showing the history of the COM.
    
    Parameters:
      gait_phases: List of gait phase dictionaries.
      torso_pos: Tuple (x, y, z) for the torso's lower corner position (x, y used).
    """
    # Create top view figure/axis on first call; reuse on subsequent calls.
    if not hasattr(self, "top_view_fig") or self.top_view_fig is None:
        self.top_view_fig = plt.figure(figsize=(8, 6))
        self.top_view_ax = self.top_view_fig.add_subplot(111)
        plt.ion()  # interactive mode on
    ax = self.top_view_ax
    fig = self.top_view_fig

    # Clear axis for update.
    ax.clear()

    # Axis limits
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top View: Planned & Adapted Footsteps and Torso COM")
    ax.grid(True)

    # Function to add a rectangle representing a foot.
    def add_foot_rect(center, foot_angle, color):
        # Use instance parameters for foot dimensions.
        l = self.foot_length
        w = self.foot_width
        # Calculate lower-left corner from center.
        lower_left = (center[0] - l/2.0, center[1] - w/2.0)
        # Convert angle from radians to degrees.
        angle_deg = np.degrees(foot_angle)
        rect = Rectangle(lower_left, l, w, angle=angle_deg, edgecolor='black',
                         facecolor=color, alpha=0.6)
        ax.add_patch(rect)

    # Draw all planned and adapted footsteps.
    for phase in gait_phases:
        # Planned left foot.
        if phase["support_leg"] in ["both", "left"]:
            center = phase["left_foot"][:2]
            # Planned footsteps use the stored foot angle (here always 0).
            add_foot_rect(center, 0, 'blue')
            # Adapted left footsteps.
            if "adapted_left_footstep" in phase:
                for step in phase["adapted_left_footstep"]:
                    add_foot_rect(step["foot_center"][:2], step.get("foot_angle", 0), 'green')
        # Planned right foot.
        if phase["support_leg"] in ["both", "right"]:
            center = phase["right_foot"][:2]
            add_foot_rect(center, 0, 'red')
            # Adapted right footsteps.
            if "adapted_right_footstep" in phase:
                for step in phase["adapted_right_footstep"]:
                    add_foot_rect(step["foot_center"][:2], step.get("foot_angle", 0), 'green')

    # Compute and plot torso COM.
    com_x = torso_pos[0] 
    com_y = torso_pos[1]
    ax.scatter(com_x, com_y, color='magenta', marker='*', s=50, label='COM')

    # Update and plot the history of the COM.
    if not hasattr(self, "com_history"):
        self.com_history = []
    self.com_history.append((com_x, com_y))
    # Unzip history points
    hx, hy = zip(*self.com_history)
    ax.plot(hx, hy, color='magenta', label='COM History')

    # Build legend based on patches and scatter.
    handles = []
    handles.append(Patch(facecolor='blue', edgecolor='black', label='Planned Left Foot'))
    handles.append(Patch(facecolor='red', edgecolor='black', label='Planned Right Foot'))
    handles.append(Patch(facecolor='green', edgecolor='black', label='Adapted Foot'))
    handles.append(plt.Line2D([], [], marker='*', linestyle='None', color='magenta', markersize=7, label='Current COM'))
    handles.append(plt.Line2D([], [], color='magenta', label='COM History'))
    ax.legend(handles=handles, loc='upper right')

    fig.canvas.draw()
    fig.canvas.flush_events()
