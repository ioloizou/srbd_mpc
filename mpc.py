import osqp
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import time



# MPC class
class MPC:
  # Constructor
  def __init__(self, mu=0.3, fz_min = 10, fz_max = 666, dt = 0.04, HORIZON_LENGTH = 10):
    # Initialize the MPC class
    self.g = -9.81 # m/s^2 Gravity
    self.robot_mass = 35 # kg
    self.dt = dt # seconds 
    self.LEGS = 2 # Number of legs
    self.HORIZON_LENGTH = HORIZON_LENGTH # Number of nodes in the MPC horizon
    self.NUM_STATES = 13  
    self.NUM_CONTROLS = 3*self.LEGS
    self.NUM_BOUNDS = 5
    # Inertia matrix of the body TO BE CHANGED
    self.INERTIA_BODY = np.diag([0.065674966, 0.053535188, 0.030808125]) # kg*m^2 torso link of g1
    # x = [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz]
    self.x = np.zeros((self.NUM_STATES, 1))
    # u = [f_left_x, f_left_y, f_left_z, f_right_x, f_right_y, f_right_z]
    self.u = np.zeros((self.NUM_CONTROLS, 1))
    # NEED TO CHANGE WEIGHTS
    # The last weight is for the gravity term
    self.q_weights = np.diag([750, 75, 1250, 8e2, 2e3, 3e4, 8e2, 2e3, 3e4, 5e2, 5e3, 5e2, 0]) # Weights from MIT humanoid orientation aware
    self.r_weights = np.diag([0.01, 0.01, 0.1, 0.01, 0.01, 0.1])
    self.mu = mu # Coefficient of friction
    self.fz_min = fz_min # Newton, Minimum normal force
    self.fz_max = fz_max # Newton, Maximum normal force


  def init_matrices(self):
    # Initialize the matrices to speed up the online computation
    self.A_continuous = np.zeros((self.NUM_STATES, self.NUM_STATES))
    self.A_discrete = np.zeros((self.NUM_STATES, self.NUM_STATES))
    self.Aqp = np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_STATES))
    self.B_continuous = np.zeros((self.NUM_STATES, self.NUM_CONTROLS))
    self.B_continuous_hor= np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS))
    self.B_discrete = np.zeros((self.NUM_STATES, self.NUM_CONTROLS))
    self.B_discrete_hor = np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS))
    self.Bqp = np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS * self.HORIZON_LENGTH))
    self.Q = np.zeros((self.NUM_STATES*self.HORIZON_LENGTH, self.NUM_STATES*self.HORIZON_LENGTH))
    self.R = np.zeros((self.NUM_CONTROLS*self.HORIZON_LENGTH, self.NUM_CONTROLS*self.HORIZON_LENGTH))
    self.Ac = np.zeros((5 * self.LEGS * self.HORIZON_LENGTH, self.NUM_CONTROLS * self.HORIZON_LENGTH))
    self.Hessian = self.Bqp.T @ self.Q @ self.Bqp + self.R
    self.x0 = np.zeros((self.NUM_STATES, 1))
    self.x_ref = np.zeros((self.NUM_STATES, 1))
    self.u_ref = np.zeros((self.NUM_CONTROLS, 1))
    self.u_min = np.zeros((self.NUM_CONTROLS, 1))
    self.u_max = np.zeros((self.NUM_CONTROLS, 1))
    self.x_min = np.zeros((self.NUM_STATES, 1))
    self.x_max = np.zeros((self.NUM_STATES, 1))
    self.u_opt = np.zeros((self.NUM_CONTROLS, self.HORIZON_LENGTH))
    self.x_opt = np.zeros((self.NUM_STATES, self.HORIZON_LENGTH))
    self.u_opt0 = np.zeros((self.NUM_CONTROLS, 1))
    self.x_opt0 = np.zeros((self.NUM_STATES, 1))

  def extract_psi(self, x_ref):
    # Extract the average yaw angle from the reference trajectory
    yaw_sum = 0
    for i in range(self.HORIZON_LENGTH):
      yaw_sum += x_ref[2, i]
    self.psi = yaw_sum / self.HORIZON_LENGTH

  def vector_to_skew_symmetric_matrix(self, v):
    # Convert a vector to a skew symmetric matrix
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
  
  def rotation_matrix_T(self, psi):
    # Rotation matrix
    self.rotation_z = np.array([[np.cos(psi), -np.sin(psi), 0], 
                                [np.sin(psi), np.cos(psi), 0], 
                                [0, 0, 1]])

  def set_Q(self):
    for i in range(self.HORIZON_LENGTH):
      self.Q[i*self.NUM_STATES:(i+1)*self.NUM_STATES, i*self.NUM_STATES:(i+1)*self.NUM_STATES] = self.Q

  def set_R(self):
    for i in range(self.HORIZON_LENGTH):
      self.R[i*self.NUM_CONTROLS:(i+1)*self.NUM_CONTROLS, i*self.NUM_CONTROLS:(i+1)*self.NUM_CONTROLS] = self.R
  
  def calculate_A_contrinuous(self):
    # Calculate the continuous A matrix
    
    # Populate the A matrix
    self.A_continuous[0:3, 6:9] = self.rotation_z
    self.A_continuous[3:6, 9:12] = np.eye(3)
    # The augmented gravity part
    self.A_continuous[11, 12] = 1

    # Assert the dimensions of the A matrix
    assert self.A_continuous.shape == (self.NUM_STATES, self.NUM_STATES)
    
    # print("A_continuous: \n", self.A_continuous)
    # print("A_continuous Shape: \n", self.A_continuous.shape)
  
  def calculate_A_discrete(self):
    # Calculate the discrete A matrix
    self.A_discrete = np.eye(12) + self.A_continuous * self.dt
    
    # print("A_discrete: \n", self.A_discrete)
    # print("A_discrete Shape: \n", self.A_discrete.shape)

    assert self.A_discrete.shape == (self.NUM_STATES, self.NUM_STATES)
    
  def calculate_B_continous(self, c_horizon, p_com_horizon):

    INERTIA_WORLD = self.rotation_z @ self.INERTIA_BODY @ self.rotation_z.T
    
    for i in range(self.HORIZON_LENGTH):
      for j in range(self.LEGS):
        # Vector from the center of mass to the contact point NEED TO CHECK THE FRAME
        r = c_horizon[i] - p_com_horizon[i]
        r_skew = self.vector_to_skew_symmetric_matrix(r)
        self.B_continuous[6:9, 3*j:(3*j+3)] = np.linalg.inv(INERTIA_WORLD) @ r_skew
        self.B_continuous[9:12, 3*j:(3*j+3)] = np.eye(3)/self.robot_mass
        assert self.B_continuous.shape == (self.NUM_STATES, self.NUM_CONTROLS)

      self.B_continuous_hor[i*self.NUM_STATES:(i+1)*self.NUM_STATES, 0:self.NUM_CONTROLS] = self.B_continuous
    assert self.B_continuous_hor.shape == (self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS)   
  
  def calculate_B_discrete(self):
    self.B_discrete_hor = self.B_continuous_hor * self.dt
    assert self.B_discrete.shape == (self.NUM_STATES, self.NUM_CONTROLS)

  def calculate_Aqp(self):
    for i in range(self.HORIZON_LENGTH):
      if i==0:
        self.Aqp[0:12, 0:12] = self.A_discrete
      else:
        self.Aqp[i*self.NUM_STATES:(i+1)*self.NUM_STATES, 0:self.NUM_STATES] = self.Aqp[(i-1)*self.NUM_STATES:i*self.NUM_STATES, 0:self.NUM_STATES] @ self.A_discrete

    assert self.Aqp.shape == (self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_STATES)
  
  def calculate_Bqp(self):
    for i in range(self.HORIZON_LENGTH):
      for j in range(i + 1):
          if i == j:
            self.Bqp[i*self.NUM_STATES:(i+1)*self.NUM_STATES, j*self.NUM_CONTROLS:(j+1)*self.NUM_CONTROLS] = self.B_discrete
          else:
            self.Bqp[i*self.NUM_STATES:(i+1)*self.NUM_STATES, j*self.NUM_CONTROLS:(j+1)*self.NUM_CONTROLS] = \
            self.Aqp[(i-j-1)*self.NUM_STATES:(i-j)*self.NUM_STATES, :] @ self.B_discrete
    
    assert self.Bqp.shape == (self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS * self.HORIZON_LENGTH)

  def calculate_Ac(self):

    # Define the block matrix (5x3)
    g_block = np.array([
      [1, 0, self.mu],
      [1, 0, -self.mu],
      [0, 1, self.mu],
      [0, 1, -self.mu],
      [0, 0, 1]
    ])

    total_rows = self.NUM_BOUNDS * self.LEGS * self.HORIZON_LENGTH
    total_cols = self.LEGS * self.HORIZON_LENGTH

    triplet_rows = []
    triplet_cols = []
    triplet_data = []

    row_offset = 0
    col_offset = 0

    while (row_offset + g_block.shape[0] <= total_rows) and (col_offset + g_block.shape[1] <= total_cols):
      for i in range(g_block.shape[0]):
        for j in range(g_block.shape[1]):
          triplet_rows.append(row_offset + i)
          triplet_cols.append(col_offset + j)
          triplet_data.append(g_block[i, j])
      row_offset += 5
      col_offset += 3

    self.Ac = coo_matrix((triplet_data, (triplet_rows, triplet_cols)), shape=(total_rows, total_cols)).tocsr()

    assert self.Ac.shape == (total_rows, total_cols)

  def calculate_bounds(self, contact):
    lower_bounds_horizon = np.zeros(self.HORIZON_LENGTH * self.LEGS * self.NUM_BOUNDS)
    upper_bounds_horizon = np.zeros(self.HORIZON_LENGTH * self.LEGS * self.NUM_BOUNDS)
      
    for horizon_step in range(self.HORIZON_LENGTH):
      lower_bounds = np.empty(self.LEGS * self.NUM_BOUNDS)
      upper_bounds = np.empty(self.LEGS * self.NUM_BOUNDS)
      for i in range(self.LEGS):
        idx = i * self.NUM_BOUNDS
        lower_bounds[idx:idx+self.NUM_BOUNDS] = np.array([
          0,                      # fx + mu*fz lower bound
          -np.inf,                # fx - mu*fz lower bound
          0,                      # fy + mu*fz lower bound
          -np.inf,                # fy - mu*fz lower bound
          self.fz_min * int(contact[horizon_step][i])
        ])
        upper_bounds[idx:idx+self.NUM_BOUNDS] = np.array([
          np.inf,                 # fx + mu*fz upper bound
          0,                      # fx - mu*fz upper bound
          np.inf,                 # fy + mu*fz upper bound
          0,                      # fy - mu*fz upper bound
          self.fz_max * int(contact[horizon_step][i])
        ])
      start = horizon_step * self.LEGS * self.NUM_BOUNDS
      end = start + self.LEGS * self.NUM_BOUNDS
      lower_bounds_horizon[start:end] = lower_bounds
      upper_bounds_horizon[start:end] = upper_bounds

    self.lower_bounds_horizon = lower_bounds_horizon
    self.upper_bounds_horizon = upper_bounds_horizon

  def calculate_hessian(self):
    self.Hessian = self.Bqp.T @ self.Q @ self.Bqp + self.R
  
  def calculate_gradient(self):
    self.gradient = self.Bqp.T @ self.Q @ (self.Aqp @ self.x0 - self.x_ref) 
    # + self.R @ self.u_ref
   

  def solve_qp(self):
    # Convert problem data to sparse matrices
    P = sparse.csc_matrix(self.Hessian)
    A = self.Ac.tocsc() if hasattr(self.Ac, "tocsc") else sparse.csc_matrix(self.Ac)
    q = self.gradient
    l = self.lower_bounds_horizon
    u = self.upper_bounds_horizon

    t0 = time.time()

    # Initialize the OSQP solver if it hasn't been created yet
    if not hasattr(self, "qp_solver"):
      self.qp_solver = osqp.OSQP()
      self.qp_solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, warm_start=True)
    else:
      # Update the OSQP problem data with new values
      self.qp_solver.update(q=q, P=P, l=l, u=u)

    t1 = time.time()
    result = self.qp_solver.solve()
    t2 = time.time()

    print("Solver init time: {:.2f}ms".format((t1-t0)*1000))
    print("Solve time: {:.2f}ms".format((t2-t1)*1000))

    return result
  
  def compute_rollout(self):
    # Compute the rollout
    for i in range(1, self.HORIZON_LENGTH):
      self.x_opt[:, i] = self.A_discrete @ self.x_opt[:, i-1] + self.B_discrete_hor[(i-1)*self.NUM_STATES:i*self.NUM_STATES, 0:self.NUM_CONTROLS] @ self.u_opt[:, i-1]
      self.u_opt[:, i] = self.u_opt[:, i-1]

    # print("x_opt: \n", self.x_opt)
    # print("u_opt: \n", self.u_opt)