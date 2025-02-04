import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

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
  Returns the list of faces for a cuboid with the given position, size, and orientation.
  
  Parameters:
    pos: Tuple (x, y, z) for the lower corner (bottom face) of the cuboid.
    size: Tuple (dx, dy, dz) giving the dimensions along x, y, and z.
    euler_angles: Tuple (roll, pitch, yaw) in radians.
  """
  x, y, z = pos
  dx, dy, dz = size
  
  # Define the eight vertices of the cuboid based on the lower corner.
  vertices = np.array([
    [x, y, z],             # 0: bottom face
    [x + dx, y, z],        # 1
    [x + dx, y + dy, z],   # 2
    [x, y + dy, z],        # 3
    [x, y, z + dz],        # 4: top face
    [x + dx, y, z + dz],   # 5
    [x + dx, y + dy, z + dz], # 6
    [x, y + dy, z + dz]     # 7
  ])

  # Define a pivot about which to rotate.
  # Here we choose the center of the cuboid (center of its faces).
  pivot = np.array([x + dx/2.0, y + dy/2.0, z + dz/2.0])
  
  # Get the rotation matrix.
  R = euler_rotation_matrix(*euler_angles)
  
  # Rotate each vertex about the pivot.
  rotated_vertices = []
  for v in vertices:
    rotated = R @ (v - pivot) + pivot
    rotated_vertices.append(rotated)
  rotated_vertices = np.array(rotated_vertices)
  
  # Define six faces using the rotated vertices.
  faces = [
    [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],  # bottom face
    [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],  # top face
    [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],  # side face 1
    [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],  # side face 2
    [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],  # side face 3
    [rotated_vertices[3], rotated_vertices[0], rotated_vertices[4], rotated_vertices[7]],  # side face 4
  ]

  cuboid = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=1.)
  return cuboid

def foot_vertices(center, length, width, angle):
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

def set_torso_pose(x, y, z, roll, pitch, yaw):
  body_pos = (x, y, z)
  euler_angles = (roll, pitch, yaw)
  return body_pos, euler_angles

def set_left_foot_pose(x, y, angle):
  center_floor = np.array([x, y])
  left_center = center_floor + np.array([0, foot_sep / 2.0])
  left_foot = foot_vertices(left_center, foot_length, foot_width, angle)
  return left_foot

def set_right_foot_pose(x, y, angle):
  center_floor = np.array([x, y])
  right_center = center_floor - np.array([0, foot_sep / 2.0])
  right_foot = foot_vertices(right_center, foot_length, foot_width, angle)
  return right_foot

def update(frame):
  ax.clear()

  # Time-dependent translation (forward motion).
  t = frame / 10.0
  x = t
  y = t
  z = 1 + 0.5*abs(np.sin(t))
  body_pos = (x, y, z)
  
  # Dynamically update x-axis limits so the robot stays in view.
  x_margin = 3
  y_margin = 3
  ax.set_xlim(x - x_margin, x + x_margin)
  ax.set_ylim(y - y_margin, y + y_margin)
  ax.set_zlim(0, 3)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('3D Humanoid with Two Feet')
  
  # Define Euler angles (roll, pitch, yaw) for the torso.
  roll = t*0.1
  pitch = t*0.1 
  yaw = t*0.1
  euler_angles = (roll, pitch, yaw)

  cuboid = get_torso_faces(body_pos, body_size, euler_angles)
  ax.add_collection3d(cuboid)

  left_foot = set_left_foot_pose(x + 0.2, y, 0)
  right_foot = set_right_foot_pose(x, y, 0)
  left_foot_poly = Poly3DCollection([left_foot], facecolors='brown', linewidths=1, edgecolors='k', alpha=0.8)
  right_foot_poly = Poly3DCollection([right_foot], facecolors='brown', linewidths=1, edgecolors='k', alpha=0.8)
  ax.add_collection3d(left_foot_poly)
  ax.add_collection3d(right_foot_poly)
  
  # Draw a floor centered around the current x position.
  floor_x = np.linspace(x - x_margin, x + x_margin, 2)
  floor_y = np.linspace(y - y_margin, y+  y_margin, 2)
  floor_x, floor_y = np.meshgrid(floor_x, floor_y)
  floor_z = np.zeros_like(floor_x)
  ax.plot_surface(floor_x, floor_y, floor_z, alpha=0.2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Humanoid with Two Feet')

body_size = (0.5, 0.3, 1.)  # width, depth, height
foot_length = 0.4
foot_width = 0.2
foot_sep = 0.4

ani = animation.FuncAnimation(fig, update, frames=200, interval=50)
plt.show()
