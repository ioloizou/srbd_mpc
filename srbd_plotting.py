import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.animation as animation

# Global parameters.
body_size = (0.3, 0.5, 1.)  # width, depth, height
foot_length = 0.4
foot_width = 0.2
foot_sep = 0.4

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

def get_torso_faces(pos, size, euler_angles=(0, 0, 0)):
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
  R = euler_rotation_matrix(*euler_angles)
  
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

def foot_vertices(center, length, width, angle):
  """
  Returns 3D vertices for a foot polygon given its center, dimensions, and rotation.
  """
  cx, cy = center
  hl = length / 2.0
  hw = width / 2.0
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

def set_left_foot_pose(x, y, angle):
  """
  Returns the vertices of the left foot.
  Computes the foot center using a predefined offset.
  """
  center_floor = np.array([x, y])
  left_first_vertice = center_floor
  # + np.array([foot_length/2., foot_width/2.])
  return foot_vertices(left_first_vertice, foot_length, foot_width, angle)

def set_right_foot_pose(x, y, angle):
  """
  Returns the vertices of the right foot.
  Computes the foot center using a predefined offset.
  """
  center_floor = np.array([x, y])
  right_first_vertice = center_floor
  # + np.array([foot_length/2., foot_width/2.])
  return foot_vertices(right_first_vertice, foot_length, foot_width, angle)

def draw_humanoid(torso_pos, torso_euler, left_foot_center, left_foot_angle, right_foot_center, right_foot_angle, is_static=True):
  """
  Updates the plot with the provided torso and feet poses.
  
  Parameters:
    torso_pos: Tuple (x, y, z) position of the torso (lower corner).
    torso_euler: Tuple (roll, pitch, yaw) in radians for the torso.
    left_foot_center: Tuple (x, y) for the left foot center.
    left_foot_angle: Rotation angle in radians for the left foot.
    right_foot_center: Tuple (x, y) for the right foot center.
    right_foot_angle: Rotation angle in radians for the right foot.
    is_static: Boolean flag. If True, the map remains a fixed 10m x 10m area,
               allowing the robot to move within a static coordinate frame.
  """
  ax.clear()
  # Set axis limits.
  if is_static:
    margin = 3  # static map extends from -5 to 5 in both x and y directions.
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
  else:
    x, y, z = torso_pos
    margin = 3
    ax.set_xlim(x - margin, x + margin)
    ax.set_ylim(y - margin, y + margin)
    
  ax.set_zlim(0, 3)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('3D Humanoid with Two Feet')
  
  # Draw torso.
  cuboid = get_torso_faces(torso_pos, body_size, torso_euler)
  ax.add_collection3d(cuboid)
  
  # Draw feet.
  left_foot = set_left_foot_pose(left_foot_center[0], left_foot_center[1], left_foot_angle)
  right_foot = set_right_foot_pose(right_foot_center[0], right_foot_center[1], right_foot_angle)
  
  left_foot_poly = Poly3DCollection([left_foot], facecolors='brown', linewidths=1, edgecolors='k', alpha=0.8)
  right_foot_poly = Poly3DCollection([right_foot], facecolors='brown', linewidths=1, edgecolors='k', alpha=0.8)

  # Draw ground reaction forces.
  left_fx = 2.
  left_fy = 2. 
  left_fz = 40.
  ax.quiver(left_foot_center[0], left_foot_center[1], 0, left_fx, left_fy, left_fz, color='r', length=0.02)
  
  right_fx = 3.
  right_fy = 1.
  right_fz = 30.
  ax.quiver(right_foot_center[0], right_foot_center[1], 0, right_fx, right_fy, right_fz, color='r', length=0.02)

  ax.add_collection3d(left_foot_poly)
  ax.add_collection3d(right_foot_poly)
  
  # Draw floor.
  if is_static:
    floor_min, floor_max = -margin, margin
  else:
    x, y, _ = torso_pos
    floor_min, floor_max = x - margin, x + margin
  
  floor_x = np.linspace(floor_min, floor_max, 2)
  floor_y = np.linspace(floor_min, floor_max, 2)
  floor_x, floor_y = np.meshgrid(floor_x, floor_y)
  floor_z = np.zeros_like(floor_x)
  ax.plot_surface(floor_x, floor_y, floor_z, alpha=0.2)

def update_and_plot_humanoid(pose=None, pause_time=0):
  """
  Updates the plot with the provided pose and displays it.
  
  Each pose is a dictionary containing the keys:
    'torso_pos': Tuple (x, y, z) for the torso (lower corner) position.
    'torso_euler': Tuple (roll, pitch, yaw) in radians for the torso.
    'left_foot_center': Tuple (x, y) for the left foot center.
    'left_foot_angle': Rotation angle in radians for the left foot.
    'right_foot_center': Tuple (x, y) for the right foot center.
    'right_foot_angle': Rotation angle in radians for the right foot.
    
  If no pose is provided (i.e. pose is None), the previous pose is reused.
  The pause_time is set to 0 (or near-zero) by default.
  """
  global fig, ax

  if 'fig' not in globals() or 'ax' not in globals():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

  draw_humanoid(
    pose['torso_pos'],
    pose['torso_euler'],
    pose['left_foot_center'], pose['left_foot_angle'],
    pose['right_foot_center'], pose['right_foot_angle']
  )
  plt.draw()
  # Use a very short pause to allow the plot to update.
  plt.pause(pause_time if pause_time > 0 else 0.0001)

def animate_humanoid_demo():
  """
  Demo animation showing time-varying torso and feet poses.
  
  For each timestep, the torso moves forward with a changing orientation,
  and the feet positions/angles are updated accordingly.
  """
  global fig, ax
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  def animate(frame):
    # Time used to create dynamic example poses.
    t = frame / 10.0
    # Torso: use t for position and small rotation increments.
    torso_pos = (t, t, 1 + 0.5 * abs(np.sin(t)))
    torso_euler = (t*0.1, t*0.1, t*0.1)
    # Feet: for demo, set foot centers relative to torso position.
    left_foot_center = (t + 0.2, t)   # you can adjust as needed
    right_foot_center = (t, t)
    left_foot_angle = 0.1 * np.sin(t)
    right_foot_angle = -0.1 * np.sin(t)
    
    draw_humanoid(torso_pos, torso_euler, left_foot_center, left_foot_angle, right_foot_center, right_foot_angle)
  
  ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)
  plt.show()

# if __name__ == "__main__":

#   plt.ion()  # enable interactive mode
  
#   for t in np.linspace(0, 10, 100):
#     demo_pose = {
#       'torso_pos': (t, t, 1 + 0.5 * abs(np.sin(t))),
#       'torso_euler': (t * 0.1, t * 0.1, t * 0.1),
#       'left_foot_center': (t + 0.2, t),
#       'left_foot_angle': 0.1 * np.sin(t),
#       'right_foot_center': (t, t),
#       'right_foot_angle': -0.1 * np.sin(t)
#     }
#     update_and_plot_humanoid(demo_pose)
  
  # animate_humanoid_demo()  # uncomment to run the animation demo
  # plt.show()  # display the final plot window
