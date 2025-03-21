#include "mpc.hpp"
#include <chrono>

namespace g1_mpc
{
  // Define static members
  const double MPC::g_ = -9.80665;
  int MPC::horizon_length_ = 15;

  MPC::MPC(double mu, double fz_min, double fz_max, double dt, int horizon_length)
      : dt_(dt),
        num_contacts_(4),
        num_states_(13),
        mu_(mu),
        fz_min_(fz_min),
        fz_max_(fz_max)
  {
    // Set the static horizon length to the provided value
    horizon_length_ = horizon_length;

    // Set number of controls (3 forces per contact)
    num_controls_ = 3 * num_contacts_;

    // Initialize the inertia matrix
    inertia_body_ << 3.20564, 4.35027e-05, 0.425526,
        4.35027e-05, 3.05015, -0.00065082,
        0.425526, -0.00065082, 0.552353;

    // Initialize state and control vectors
    x_ = Eigen::VectorXd::Zero(num_states_);
    u_ = Eigen::VectorXd::Zero(num_controls_);

    // Set up Q weights - Standing double support weights
    Eigen::VectorXd q_diag(num_states_);
    q_diag << 7e5, 7e4, 1e4,
        5e5, 5e5, 3e6,
        3e3, 3e3, 3e3,
        5e3, 1e3, 1e4, 0;
    Q_weights_ = q_diag.asDiagonal();

    // Set up R weights
    Eigen::VectorXd r_diag = Eigen::VectorXd::Constant(num_controls_, 0.001);
    R_weights_ = r_diag.asDiagonal();

    // Initialize all matrices for the solver
    initMatrices();
  }

  MPC::~MPC()
  {
    // Clean up OSQP solver resources
    if (qp_solver_)
    {
      osqp_cleanup(qp_solver_);
    }
    if (data_)
    {
      if (data_->A)
        c_free(data_->A);
      if (data_->P)
        c_free(data_->P);
      c_free(data_);
    }
    if (settings_)
    {
      c_free(settings_);
    }
  }

  void MPC::initMatrices()
  {
    // Initialize all matrices with correct sizes to avoid reallocations
    A_continuous_ = Eigen::MatrixXd::Zero(num_states_, num_states_);
    A_discrete_ = Eigen::MatrixXd::Zero(num_states_, num_states_);
    Aqp_ = Eigen::MatrixXd::Zero(num_states_ * horizon_length_, num_states_);

    B_continuous_ = Eigen::MatrixXd::Zero(num_states_, num_controls_);
    B_continuous_hor_ = Eigen::MatrixXd::Zero(num_states_ * horizon_length_, num_controls_);
    B_discrete_ = Eigen::MatrixXd::Zero(num_states_, num_controls_);
    B_discrete_hor_ = Eigen::MatrixXd::Zero(num_states_ * horizon_length_, num_controls_);
    Bqp_ = Eigen::MatrixXd::Zero(num_states_ * horizon_length_, num_controls_ * horizon_length_);

    Q_ = Eigen::MatrixXd::Zero(num_states_ * horizon_length_, num_states_ * horizon_length_);
    R_ = Eigen::MatrixXd::Zero(num_controls_ * horizon_length_, num_controls_ * horizon_length_);

    Hessian_ = Eigen::MatrixXd::Zero(num_controls_ * horizon_length_, num_controls_ * horizon_length_);
    rotation_z_T_ = Eigen::Matrix3d::Zero();

    x0_ = Eigen::VectorXd::Zero(num_states_);
    gradient_ = Eigen::VectorXd::Zero(num_controls_ * horizon_length_);

    x_ref_hor_ = Eigen::MatrixXd::Zero(horizon_length_, num_states_);
    u_ref_hor_ = Eigen::MatrixXd::Zero(horizon_length_, num_controls_);
    u_opt_ = Eigen::MatrixXd::Zero(horizon_length_, num_controls_);
    x_opt_ = Eigen::MatrixXd::Zero(horizon_length_, num_states_);
    u_opt0_ = Eigen::VectorXd::Zero(num_controls_);

    // Allocation for temporary vectors
    x_ref_hor_vec_ = Eigen::VectorXd::Zero(num_states_ * horizon_length_);
    r_ = Eigen::Vector3d::Zero();

    // Preallocate contact and p_com horizon matrices
    c_horizon_ = Eigen::MatrixXd::Zero(horizon_length_, num_contacts_ * 3);
    p_com_horizon_ = Eigen::MatrixXd::Zero(horizon_length_, 3);
  }

  double MPC::extractPsi()
  {
    double yaw_sum = 0;
    for (int i = 0; i < horizon_length_; i++)
    {
      yaw_sum += x_ref_hor_(i, 2);
    }
    psi_ = yaw_sum / horizon_length_;
    return psi_;
  }

  Eigen::Matrix3d MPC::vectorToSkewSymmetricMatrix(const Eigen::Vector3d &v)
  {
    Eigen::Matrix3d skew;
    skew << 0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return skew;
  }

  void MPC::calculateRotationMatrixT()
  {
    rotation_z_T_ << cos(psi_), sin(psi_), 0,
        -sin(psi_), cos(psi_), 0,
        0, 0, 1;
  }

  void MPC::setQ()
  {
    for (int i = 0; i < horizon_length_; i++)
    {
      Q_.block(i * num_states_, i * num_states_, num_states_, num_states_) = Q_weights_;
    }
  }

  void MPC::setR()
  {
    for (int i = 0; i < horizon_length_; i++)
    {
      R_.block(i * num_controls_, i * num_controls_, num_controls_, num_controls_) = R_weights_;
    }
  }

  void MPC::calculateAContinuous()
  {
    // Reset the matrix to zero first
    A_continuous_.setZero();

    // Upper-right 3x3 block: rotation matrix
    A_continuous_.block(0, 6, 3, 3) = rotation_z_T_.transpose();

    // Middle-right 3x3 block: identity matrix
    A_continuous_.block(3, 9, 3, 3) = Eigen::Matrix3d::Identity();

    // Augmented gravity part
    A_continuous_(11, 12) = 1;
  }

  void MPC::calculateADiscrete()
  {
    A_discrete_ = Eigen::MatrixXd::Identity(num_states_, num_states_) + A_continuous_ * dt_;
  }

  void MPC::calculateBContinuous(const std::vector<std::vector<double>> &c_horizon,
                                 const std::vector<std::vector<double>> &p_com_horizon)
  {
    // Convert input vectors to Eigen matrices for efficient processing
    for (int i = 0; i < horizon_length_; i++)
    {
      for (int j = 0; j < num_contacts_ * 3; j++)
      {
        c_horizon_(i, j) = c_horizon[i][j];
      }
      for (int j = 0; j < num_states_; j++)
      {
        p_com_horizon_(i, j) = p_com_horizon[i][j];
      }
    }

    // Calculate world inertia
    Eigen::Matrix3d inertia_world = rotation_z_T_ * inertia_body_ * rotation_z_T_.transpose();
    Eigen::Matrix3d inertia_world_inv = inertia_world.inverse();

    // For each horizon step
    for (int i = 0; i < horizon_length_; i++)
    {
      B_continuous_.setZero();

      for (int j = 0; j < num_contacts_; j++)
      {
        // Extract contact point and COM position
        Eigen::Vector3d contact_pos(c_horizon_(i, 3 * j), c_horizon_(i, 3 * j + 1), c_horizon_(i, 3 * j + 2));
        Eigen::Vector3d com_pos(p_com_horizon_(i, 3), p_com_horizon_(i, 4), p_com_horizon_(i, 5));

        // Vector from COM to contact point
        r_ = contact_pos - com_pos;
        Eigen::Matrix3d r_skew = vectorToSkewSymmetricMatrix(r_);

        // Fill B matrix for this contact
        B_continuous_.block(6, 3 * j, 3, 3) = inertia_world_inv * r_skew;
        B_continuous_.block(9, 3 * j, 3, 3) = Eigen::Matrix3d::Identity() / robot_mass_;
      }

      // Store in the full horizon B matrix
      B_continuous_hor_.block(i * num_states_, 0, num_states_, num_controls_) = B_continuous_;
    }
  }

  void MPC::calculateBDiscrete()
  {
    for (int i = 0; i < horizon_length_; i++)
    {
      Eigen::MatrixXd current_B = B_continuous_hor_.block(i * num_states_, 0, num_states_, num_controls_);
      B_discrete_ = current_B * dt_;
      B_discrete_hor_.block(i * num_states_, 0, num_states_, num_controls_) = B_discrete_;
    }
  }

  void MPC::calculateAqp()
  {
    for (int i = 0; i < horizon_length_; i++)
    {
      if (i == 0)
      {
        Aqp_.block(0, 0, num_states_, num_states_) = A_discrete_;
      }
      else
      {
        Aqp_.block(i * num_states_, 0, num_states_, num_states_) =
            Aqp_.block((i - 1) * num_states_, 0, num_states_, num_states_) * A_discrete_;
      }
    }
  }

  void MPC::calculateBqp()
  {
    for (int i = 0; i < horizon_length_; i++)
    {
      for (int j = 0; j <= i; j++)
      {
        if (i == j)
        {
          Bqp_.block(i * num_states_, j * num_controls_, num_states_, num_controls_) =
              B_discrete_hor_.block(j * num_states_, 0, num_states_, num_controls_);
        }
        else
        {
          Bqp_.block(i * num_states_, j * num_controls_, num_states_, num_controls_) =
              Aqp_.block((i - j - 1) * num_states_, 0, num_states_, num_states_) *
              B_discrete_hor_.block(j * num_states_, 0, num_states_, num_controls_);
        }
      }
    }
  }

  void MPC::calculateAc()
  {
    // Create the constraint matrix for friction cone
    Eigen::MatrixXd g_block(num_bounds_, num_controls_/num_contacts_);
    g_block << 1, 0, mu_,
        1, 0, -mu_,
        0, 1, mu_,
        0, 1, -mu_,
        0, 0, 1;

    // Build the full constraint matrix using sparse blocks
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_bounds_ * num_controls_ * horizon_length_);

    for (int h = 0; h < horizon_length_; h++)
    {
      for (int c = 0; c < num_contacts_; c++)
      {
        for (int i = 0; i < num_bounds_; i++)
        {
          for (int j = 0; j < 3; j++)
          {
            if (g_block(i, j) != 0)
            {
              int row = h * num_contacts_ * num_bounds_ + c * num_bounds_ + i;
              int col = h * num_controls_ + c * 3 + j;
              tripletList.push_back(T(row, col, g_block(i, j)));
            }
          }
        }
      }
    }

    Ac_.resize(num_bounds_ * num_contacts_ * horizon_length_, num_controls_ * horizon_length_);
    Ac_.setFromTriplets(tripletList.begin(), tripletList.end());
  }

  void MPC::calculateBounds(const std::vector<std::vector<int>> &contact)
  {
    std::vector<double> lower_bounds_list;
    std::vector<double> upper_bounds_list;

    lower_bounds_list.reserve(horizon_length_ * num_contacts_ * num_bounds_);
    upper_bounds_list.reserve(horizon_length_ * num_contacts_ * num_bounds_);

    for (int horizon_step = 0; horizon_step < horizon_length_; horizon_step++)
    {
      for (int leg = 0; leg < num_contacts_; leg++)
      {
        int contact_active = contact[horizon_step][leg];

        // Lower bounds
        lower_bounds_list.push_back(0);                        // fx + mu*fz
        lower_bounds_list.push_back(-INFINITY);                // fx - mu*fz
        lower_bounds_list.push_back(0);                        // fy + mu*fz
        lower_bounds_list.push_back(-INFINITY);                // fy - mu*fz
        lower_bounds_list.push_back(fz_min_ * contact_active); // fz_min

        // Upper bounds
        upper_bounds_list.push_back(INFINITY);                 // fx + mu*fz
        upper_bounds_list.push_back(0);                        // fx - mu*fz
        upper_bounds_list.push_back(INFINITY);                 // fy + mu*fz
        upper_bounds_list.push_back(0);                        // fy - mu*fz
        upper_bounds_list.push_back(fz_max_ * contact_active); // fz_max
      }
    }

    // Convert to Eigen vectors
    lower_bounds_horizon_ = Eigen::Map<Eigen::VectorXd>(lower_bounds_list.data(), lower_bounds_list.size());
    upper_bounds_horizon_ = Eigen::Map<Eigen::VectorXd>(upper_bounds_list.data(), upper_bounds_list.size());
  }

  void MPC::calculateHessian()
  {
    Hessian_ = Bqp_.transpose() * Q_ * Bqp_ + R_;
  }

  void MPC::calculateGradient()
  {
    // Reshape x_ref_hor to vector for gradient computation
    x_ref_hor_vec_ = Eigen::Map<Eigen::VectorXd>(x_ref_hor_.data(), num_states_ * horizon_length_);

    // Calculate gradient
    gradient_ = Bqp_.transpose() * Q_ * ((Aqp_ * x0_) - x_ref_hor_vec_);
  }

  OSQPWorkspace *MPC::solveQP()
  {
    // Convert Eigen matrices to OSQP sparse matrices
    // P matrix (Hessian) should be upper triangular for OSQP
    Eigen::MatrixXd H_upper = Hessian_.triangularView<Eigen::Upper>();
    Eigen::SparseMatrix<double> P_sparse = H_upper.sparseView();
    
    csc *P_csc = csc_matrix(P_sparse.rows(), P_sparse.cols(),
                            P_sparse.nonZeros(), P_sparse.valuePtr(),
                            P_sparse.innerIndexPtr(), P_sparse.outerIndexPtr());

    // A matrix (constraints)
    csc *A_csc = csc_matrix(Ac_.rows(), Ac_.cols(),
                            Ac_.nonZeros(), Ac_.valuePtr(),
                            Ac_.innerIndexPtr(), Ac_.outerIndexPtr());

    // Clean up previous data if exists
    if (data_)
    {
      if (data_->A)
        c_free(data_->A);
      if (data_->P)
        c_free(data_->P);
      c_free(data_);
    }

    
    // Set up data
    data_ = (OSQPData *)c_malloc(sizeof(OSQPData));
    data_->n = num_controls_ * horizon_length_;
    data_->m = num_bounds_ * num_contacts_ * horizon_length_;
    data_->P = P_csc;
    data_->A = A_csc;
    data_->q = gradient_.data();
    data_->l = lower_bounds_horizon_.data();
    data_->u = upper_bounds_horizon_.data();
    
    // Set up settings if not already set
    if (!settings_)
    {
      settings_ = new OSQPSettings();
      osqp_set_default_settings(settings_);
      settings_->verbose = true;
      settings_->warm_start = true;
    }
    
    // Setup solver
    auto t_start = std::chrono::high_resolution_clock::now();
    
    if (qp_solver_)
    {
      osqp_cleanup(qp_solver_);
    }
    
    osqp_setup(&qp_solver_, data_, settings_);
    
    auto t_setup = std::chrono::high_resolution_clock::now();
    
    // Solve problem
    osqp_solve(qp_solver_);
    
    auto t_solve = std::chrono::high_resolution_clock::now();
    
    // Extract solution
    for (int i = 0; i < horizon_length_; i++)
    {
      for (int j = 0; j < num_controls_; j++)
      {
        u_opt_(i, j) = qp_solver_->solution->x[i * num_controls_ + j];
      }
    }
    
    // Debug timing info
    auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(t_setup - t_start).count() / 1000.0;
    auto solve_time = std::chrono::duration_cast<std::chrono::microseconds>(t_solve - t_setup).count() / 1000.0;

    // std::cout << "Solver setup time: " << setup_time << "ms" << std::endl;
    // std::cout << "Solve time: " << solve_time << "ms" << std::endl;

    return qp_solver_;
  }

  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> MPC::update(
      const std::vector<std::vector<int>> &contact,
      const std::vector<std::vector<double>> &c_horizon,
      const std::vector<std::vector<double>> &p_com_horizon,
      const Eigen::VectorXd *x_current,
      bool one_rollout)
  {
    extractPsi();
    calculateRotationMatrixT();
    setQ();
    setR();
    calculateAContinuous();
    calculateADiscrete();
    calculateBContinuous(c_horizon, p_com_horizon);
    calculateBDiscrete();
    calculateAqp();
    calculateBqp();
    calculateAc();
    calculateBounds(contact);
    calculateHessian();
    calculateGradient();
    solveQP();
    computeRollout(x_current, one_rollout);

    return {u_opt_, x_opt_};
  }

  void MPC::computeRollout(const Eigen::VectorXd *x_current, bool only_first_step)
  {
    // Use provided initial state if available
    if (x_current)
    {
      x_opt_.row(0) = *x_current;
    }

    // Compute state rollout
    for (int i = 1; i < horizon_length_; i++)
    {
      x_opt_.row(i) = A_discrete_ * x_opt_.row(i - 1).transpose() +
                      B_discrete_hor_.block((i - 1) * num_states_, 0, num_states_, num_controls_) *
                          u_opt_.row(i - 1).transpose();

      if (only_first_step)
      {
        return;
      }
    }

    // Debug output
    // std::cout << "x_opt: \n" << x_opt_ << std::endl;
    // std::cout << "u_opt: \n" << u_opt_ << std::endl;

    // Calculate total force on robot
    Eigen::MatrixXd total_force = u_opt_.rowwise().sum();
    // std::cout << "Total force on the robot: \n" << total_force << std::endl;
  }

  void MPC::debugPrintMatrixDimensions() const
  {
	  std::cout << "--- Matrix and vector dimensions ---" << std::endl;

	  std::cout << "x_ (state vector): " << x_.size() << std::endl;
	  std::cout << "u_ (control vector): " << u_.size() << std::endl;

	  std::cout << "A_continuous_: " << A_continuous_.rows() << " x " << A_continuous_.cols() << std::endl;
	  std::cout << "A_discrete_: " << A_discrete_.rows() << " x " << A_discrete_.cols() << std::endl;
	  std::cout << "Aqp_: " << Aqp_.rows() << " x " << Aqp_.cols() << std::endl;

	  std::cout << "B_continuous_: " << B_continuous_.rows() << " x " << B_continuous_.cols() << std::endl;
	  std::cout << "B_continuous_hor_: " << B_continuous_hor_.rows() << " x " << B_continuous_hor_.cols() << std::endl;
	  std::cout << "B_discrete_: " << B_discrete_.rows() << " x " << B_discrete_.cols() << std::endl;
	  std::cout << "B_discrete_hor_: " << B_discrete_hor_.rows() << " x " << B_discrete_hor_.cols() << std::endl;
	  std::cout << "Bqp_: " << Bqp_.rows() << " x " << Bqp_.cols() << std::endl;

	  std::cout << "Q_: " << Q_.rows() << " x " << Q_.cols() << std::endl;
	  std::cout << "R_: " << R_.rows() << " x " << R_.cols() << std::endl;
	  std::cout << "Hessian_: " << Hessian_.rows() << " x " << Hessian_.cols() << std::endl;

	  std::cout << "rotation_z_T_: " << rotation_z_T_.rows() << " x " << rotation_z_T_.cols() << std::endl;
	  
	  std::cout << "x0_ (initial state vector): " << x0_.size() << std::endl;
	  std::cout << "gradient_: " << gradient_.size() << std::endl;

	  std::cout << "x_ref_hor_: " << x_ref_hor_.rows() << " x " << x_ref_hor_.cols() << std::endl;
	  std::cout << "u_ref_hor_: " << u_ref_hor_.rows() << " x " << u_ref_hor_.cols() << std::endl;
	  std::cout << "u_opt_: " << u_opt_.rows() << " x " << u_opt_.cols() << std::endl;
	  std::cout << "x_opt_: " << x_opt_.rows() << " x " << x_opt_.cols() << std::endl;
	  std::cout << "u_opt0_: " << u_opt0_.size() << std::endl;

	  std::cout << "x_ref_hor_vec_: " << x_ref_hor_vec_.size() << std::endl;
	  std::cout << "r_: " << r_.size() << std::endl;

	  std::cout << "c_horizon_: " << c_horizon_.rows() << " x " << c_horizon_.cols() << std::endl;
	  std::cout << "p_com_horizon_: " << p_com_horizon_.rows() << " x " << p_com_horizon_.cols() << std::endl;
	  std::cout << "Ac_ (constraint matrix): " << Ac_.rows() << " x " << Ac_.cols() << std::endl;

	  std::cout << "lower_bounds_horizon_: " << lower_bounds_horizon_.size() << std::endl;
	  std::cout << "upper_bounds_horizon_: " << upper_bounds_horizon_.size() << std::endl;
	  
	  std::cout << "--------------------------------------" << std::endl;
  }

} // namespace g1_mpc
