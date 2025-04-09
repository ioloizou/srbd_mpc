#include "mpc.hpp"
#include <chrono>

namespace g1_mpc
{
  // Define static members
  const double MPC::g_ = -9.80665;
  int MPC::horizon_length_ = 10;

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

    // // Initialize the inertia matrix NEED TO CHECK WHICH IS CORRECT
    inertia_body_ << 3.20564e-1, 4.35027e-06, 0.425526e-1,
        4.35027e-06, 3.05015e-1, -0.00065082e-1,
        0.425526e-1, -0.00065082e-1, 0.552353e-1;

    // Initialize the inertia matrix
    // inertia_body_ << 3.20564, 4.35027e-05, 0.425526,
    // 4.35027e-05, 3.05015, -0.00065082,
    // 0.425526, -0.00065082, 0.552353;

    // Initialize state and control vectors
    x_ = Eigen::VectorXd::Zero(num_states_);
    u_ = Eigen::VectorXd::Zero(num_controls_);

    // MIT
    // Eigen::VectorXd q_diag(num_states_);
    // q_diag << 75e1, 75e0, 125e1,
    //       8e2, 2e3, 3e4,
    //       8e2, 2e3, 3e4,
    //       5e2, 5e3, 5e2,
    //       0;
    // Q_weights_ = q_diag.asDiagonal();

    // Eigen::VectorXd r_diag = Eigen::VectorXd::Constant(num_controls_, 0.001);
    // R_weights_ = r_diag.asDiagonal();

    // // Set up Q weights - Standing double support weights
    // Eigen::VectorXd q_diag(num_states_);
    // q_diag << 7e5, 7e4, 1e4,
    //     5e5, 5e5, 3e6,
    //     3e3, 3e3, 3e3,
    //     5e3, 1e3, 1e4, 0;
    // Q_weights_ = q_diag.asDiagonal();

    // // Set up R weights
    // Eigen::VectorXd r_diag = Eigen::VectorXd::Constant(num_controls_, 0.001);
    // R_weights_ = r_diag.asDiagonal();

    // // Set up Q weights - Standing double support weights
    // Eigen::VectorXd q_diag(num_states_);
    // q_diag << 7e5, 7e4, 1e4,
    //     5e5, 5e5, 3e6,
    //     3e3, 3e3, 3e3,
    //     5e3, 1e3, 1e4, 0;
    // Q_weights_ = q_diag.asDiagonal();

    // // Set up R weights
    // Eigen::VectorXd r_diag = Eigen::VectorXd::Constant(num_controls_, 0.001);
    // R_weights_ = r_diag.asDiagonal();

    // // // Walking in place weights
    // Eigen::VectorXd q_diag(num_states_);
    // q_diag << 2e3, 9e3, 5e2, 
    //           3e3, 2e4, 2e4, 
    //           5e2, 5e2, 1e1, 
    //           1e1, 9e2, 1e1, 0;
    // Q_weights_ = q_diag.asDiagonal();

    // Eigen::VectorXd r_diag = Eigen::VectorXd::Constant(num_controls_, 0.001);
    // R_weights_ = r_diag.asDiagonal();

    // Test weights
    Eigen::VectorXd q_diag(num_states_);
    q_diag << 8e1, 9e1, 5e1, 
              1e5, 1e5, 1e5, 
              5e2, 5e2, 1e2, 
              5e2, 9e2, 9e2, 0;
    Q_weights_ = q_diag.asDiagonal();

    Eigen::VectorXd r_diag = Eigen::VectorXd::Constant(num_controls_, 0.001);
    R_weights_ = r_diag.asDiagonal();

    // Initialize all matrices for the solver
    initMatrices();
  }

  MPC::~MPC()
  {
    
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

    Ac_ = Eigen::SparseMatrix<double>(num_bounds_ * num_contacts_ * horizon_length_, num_controls_ * horizon_length_);

    lower_bounds_horizon_ = Eigen::VectorXd::Zero(num_bounds_ * num_contacts_ * horizon_length_);
    upper_bounds_horizon_ = Eigen::VectorXd::Zero(num_bounds_ * num_contacts_ * horizon_length_);

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
    contact_horizon_ = Eigen::MatrixXd::Zero(horizon_length_, num_contacts_ * 3);
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

  void MPC::calculateBContinuous(const Eigen::MatrixXd &c_horizon,
                                 const Eigen::MatrixXd &p_com_horizon)
  {
    c_horizon_ = c_horizon;
    p_com_horizon_ = p_com_horizon;

    // Calculate world inertia
    Eigen::Matrix3d inertia_world = rotation_z_T_ * inertia_body_ * rotation_z_T_.transpose();
    Eigen::Matrix3d inertia_world_inv = inertia_world.inverse();

    // For each horizon step
    for (int i = 0; i < horizon_length_; i++)
    {
      for (int j = 0; j < num_contacts_; j++)
      {
        // Extract contact point and COM position
        Eigen::Vector3d contact_pos(c_horizon_(i, 3 * j), c_horizon_(i, 3 * j + 1), c_horizon_(i, 3 * j + 2));
        Eigen::Vector3d com_pos(p_com_horizon_(i, 0), p_com_horizon_(i, 1), p_com_horizon_(i, 2));

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
    g_block.resize(num_bounds_ , num_controls_/num_contacts_);
    g_block << 1,0,mu_,  
               1,0,-mu_, 
               0,1,mu_,  
               0,1,-mu_, 
               0,0,1;   

    std::vector<Eigen::Triplet<double>> tripletList;
    
    tripletList.reserve(9 * 40);  

    int row_offset = 0;
    int col_offset = 0;

    while (row_offset + g_block.rows() <= Ac_.rows() && col_offset + g_block.cols() <= Ac_.cols())
    {
        for (int i = 0; i < g_block.rows(); ++i)
        {
            for (int j = 0; j < g_block.cols(); ++j)
            {
                tripletList.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, g_block(i, j)));
            }
        }            
        row_offset += 5; 
        col_offset += 3; 
    }

    Ac_.setFromTriplets(tripletList.begin(), tripletList.end());
    Ac_.makeCompressed();
  }

  void MPC::calculateBounds(const Eigen::MatrixXd &contact)
  {
    contact_horizon_ = contact;
    Eigen::VectorXd lower_bounds(num_bounds_ * num_contacts_);
    Eigen::VectorXd upper_bounds(num_bounds_ * num_contacts_);
    
    int horizon_step = 0;
    
    while (horizon_step < horizon_length_)
    {
        for (int i=0; i<num_contacts_; i++)
        {
        lower_bounds.segment(i*num_bounds_, 5) << 0,                                        
                                                -std::numeric_limits<double>::infinity(),  
                                                 0,                                        
                                                -std::numeric_limits<double>::infinity(),  
                                                fz_min_*contact_horizon_(horizon_step, i);                   
                                                 
        upper_bounds.segment(i*num_bounds_, 5) << std::numeric_limits<double>::infinity(),  
                                                 0,                                        
                                                 std::numeric_limits<double>::infinity(),  
                                                 0,                                        
                                                 fz_max_*contact_horizon_(horizon_step, i);                   
        }

        lower_bounds_horizon_.segment(horizon_step*num_bounds_*num_contacts_, num_bounds_ * num_contacts_) = lower_bounds;
        upper_bounds_horizon_.segment(horizon_step*num_bounds_*num_contacts_, num_bounds_ * num_contacts_) = upper_bounds;
        horizon_step += 1;
    }
  }

  void MPC::calculateHessian()
  {
    Hessian_ = 2*(Bqp_.transpose() * Q_ * Bqp_ + R_);
    // Convert to sparse matrix for OSQP
    Hessian_sparse_ = Hessian_.sparseView();
  }

  void MPC::calculateGradient()
  {
    // Reshape x_ref_hor to vector for gradient computation
    for (int i = 0; i < horizon_length_; i++){
      x_ref_hor_vec_.segment(i*num_states_, num_states_) = x_ref_hor_.row(i).transpose();
    }
    // std::cout << "x_ref_hor_vec_: " << x_ref_hor_vec_ << std::endl;

    // Calculate gradient
    gradient_ = 2*Bqp_.transpose() * Q_ * ((Aqp_ * x0_) - x_ref_hor_vec_);
  }

  Eigen::VectorXd MPC::solveQP(){
    //Instantiate the solver
    OsqpEigen::Solver solver;

    auto t0 = std::chrono::high_resolution_clock::now();
    //Configure the solver
    if (!solver.isInitialized()){
        solver.settings()->setVerbosity(false);
        solver.settings()->setWarmStart(true);
        solver.data()->setNumberOfVariables(num_controls_*horizon_length_);
        solver.data()->setNumberOfConstraints(num_bounds_*num_contacts_*horizon_length_);
        solver.data()->setLinearConstraintsMatrix(Ac_);
        solver.data()->setHessianMatrix(Hessian_sparse_);
        solver.data()->setGradient(gradient_);
        solver.data()->setLowerBound(lower_bounds_horizon_);
        solver.data()->setUpperBound(upper_bounds_horizon_);
        solver.initSolver();
        }
    else{
        solver.updateGradient(gradient_);
        solver.updateHessianMatrix(Hessian_sparse_);
        solver.updateBounds(lower_bounds_horizon_, upper_bounds_horizon_);
        }

    //Init and solve keeping track of time at each step

    auto t1 = std::chrono::high_resolution_clock::now();
    solver.solveProblem();
    auto t2 = std::chrono::high_resolution_clock::now();

    //Process and print time intervals of the solver
    std::chrono::duration<double, std::milli> init_duration = t1 - t0;
    std::chrono::duration<double, std::milli> solve_duration = t2 - t1;
    // std::cout << "Solver init time: " << init_duration.count() << "ms" << std::endl; 
    // std::cout << "Solve time: " << solve_duration.count() << "ms" << std::endl;

    Eigen::VectorXd result = solver.getSolution();

    // Load result into u_opt_
    for (int i = 0; i < horizon_length_; i++)
    {
      u_opt_.row(i) = result.segment(i * num_controls_, num_controls_);
    }

    u_opt0_ = u_opt_.row(0);

    return result;
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
  }

  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> MPC::updateMPC(
      const Eigen::MatrixXd &contact,
      const Eigen::MatrixXd &c_horizon,
      const Eigen::MatrixXd &p_com_horizon,
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
