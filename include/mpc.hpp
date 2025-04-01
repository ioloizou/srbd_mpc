#pragma once

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <chrono>
#include <iostream>
#include <memory>

namespace g1_mpc
{

	class MPC
	{
	public:
		static const double g_; // m/s^2 Gravity
		static int horizon_length_;

		// Constructor
		MPC(double mu = 0.3, double fz_min = 10.0, double fz_max = 1000.0, double dt = 0.04, int horizon_length = 10);

		// Destructor
		~MPC();

		// Initialize matrices for performance
		void initMatrices();

		// Extract average yaw angle from reference trajectory
		double extractPsi();

		// Convert vector to skew symmetric matrix
		Eigen::Matrix3d vectorToSkewSymmetricMatrix(const Eigen::Vector3d &v);

		// Calculate rotation matrix T
		void calculateRotationMatrixT();

		// Set the Q weights matrix
		void setQ();

		// Set the R weights matrix
		void setR();

		// Calculate continuous A matrix
		void calculateAContinuous();

		// Calculate discrete A matrix
		void calculateADiscrete();

		// Calculate continuous B matrix
		void calculateBContinuous(const Eigen::MatrixXd &c_horizon,
								  const Eigen::MatrixXd &p_com_horizon);

		// Calculate discrete B matrix
		void calculateBDiscrete();

		// Calculate Aqp matrix
		void calculateAqp();

		// Calculate Bqp matrix
		void calculateBqp();

		// Calculate Ac constraint matrix
		void calculateAc();

		// Calculate bounds for the QP
		void calculateBounds(const Eigen::MatrixXd &contact);

		// Calculate Hessian matrix
		void calculateHessian();

		// Calculate gradient vector
		void calculateGradient();

		// Solve the QP problem
		Eigen::VectorXd solveQP();

		// Main update function
		std::pair<Eigen::MatrixXd, Eigen::MatrixXd> updateMPC(
			const Eigen::MatrixXd &contact,
			const Eigen::MatrixXd &c_horizon,
			const Eigen::MatrixXd &p_com_horizon,
			const Eigen::VectorXd *x_current = nullptr,
			bool one_rollout = false);

		// Compute state rollout
		void computeRollout(const Eigen::VectorXd *x_current = nullptr, bool only_first_step = false);

		// Getter and setter methods for x0_
		const Eigen::VectorXd& getX0() const { return x0_; }
		void setX0(const Eigen::VectorXd& x0) { x0_ = x0; }

		// Getter and setter methods for x_ref_hor_
		const Eigen::MatrixXd& getXRefHor() const { return x_ref_hor_; }
		void setXRefHor(const Eigen::MatrixXd& x_ref_hor) { x_ref_hor_ = x_ref_hor; }

		// Getter and setter methods for u_opt_
		const Eigen::MatrixXd& getUOpt() const { return u_opt_; }
		void setUOpt(const Eigen::MatrixXd& u_opt) { u_opt_ = u_opt; }

		// Getter and setter methods for x_opt_
		const Eigen::MatrixXd& getXOpt() const { return x_opt_; }
		void setXOpt(const Eigen::MatrixXd& x_opt) { x_opt_ = x_opt; }

		// Getter and setter methods for u_opt0_
		const Eigen::VectorXd& getUOpt0() const { return u_opt0_; }
		void setUOpt0(const Eigen::VectorXd& u_opt0) { u_opt0_ = u_opt0; }

		// Getter and setter methods for r_
		const Eigen::Vector3d& getr() const { return r_; }
		void setr(const Eigen::Vector3d& r) { r_ = r; }
		
		// Getter and setter methods for contact_horizon_
		const Eigen::MatrixXd& getContactHorizon() const { return contact_horizon_; }
		void setContactHorizon(const Eigen::MatrixXd& contact_horizon) { contact_horizon_ = contact_horizon; }
			
		// Getter and setter methods for p_com_horizon_
		const Eigen::MatrixXd& getPComHorizon() const { return p_com_horizon_; }
		void setPComHorizon(const Eigen::MatrixXd& p_com_horizon) { p_com_horizon_ = p_com_horizon; }
		
		// Getter and setter methods for p_com_horizon_
		const Eigen::MatrixXd& getCHorizon() const { return c_horizon_; }
		void setCHorizon(const Eigen::MatrixXd& c_horizon) { c_horizon_ = c_horizon; }
		
		// Debugging function to print matrix dimensions
		void debugPrintMatrixDimensions() const;
	private:
		// Constants
		const double robot_mass_ = 34.13385728; // kg
		double dt_;								// seconds
		int num_contacts_;						// Number of contacts
		int num_states_;
		int num_controls_;
		const int num_bounds_ = 5;

		// Inertia matrix
		Eigen::Matrix3d inertia_body_;

		// State and control vectors
		Eigen::VectorXd x_;
		Eigen::VectorXd u_;

		// Weight matrices
		Eigen::DiagonalMatrix<double, Eigen::Dynamic> Q_weights_;
		Eigen::DiagonalMatrix<double, Eigen::Dynamic> R_weights_;

		// Friction and force limits
		double mu_;
		double fz_min_;
		double fz_max_;
		double psi_; // Extracted yaw angle

		// Matrices
		Eigen::MatrixXd A_continuous_;
		Eigen::MatrixXd A_discrete_;
		Eigen::MatrixXd Aqp_;

		Eigen::MatrixXd B_continuous_;
		Eigen::MatrixXd B_continuous_hor_;
		Eigen::MatrixXd B_discrete_;
		Eigen::MatrixXd B_discrete_hor_;
		Eigen::MatrixXd Bqp_;

		Eigen::MatrixXd Q_;
		Eigen::MatrixXd R_;
		Eigen::SparseMatrix<double> Ac_;
		Eigen::MatrixXd Hessian_;
		Eigen::SparseMatrix<double> Hessian_sparse_;
		Eigen::Matrix3d rotation_z_T_;

		// Vectors for QP
		Eigen::VectorXd x0_;
		Eigen::VectorXd gradient_;
		Eigen::VectorXd lower_bounds_horizon_;
		Eigen::VectorXd upper_bounds_horizon_;

		// Reference and solution vectors
		Eigen::MatrixXd x_ref_hor_;
		Eigen::VectorXd x_ref_hor_vec_;
		Eigen::MatrixXd u_ref_hor_;
		Eigen::MatrixXd u_opt_;
		Eigen::MatrixXd x_opt_;
		Eigen::VectorXd u_opt0_;

		// Temp vectors
		Eigen::Vector3d r_;

		// Matrix for storing contact and p_com info
		Eigen::MatrixXd contact_horizon_;
		Eigen::MatrixXd c_horizon_;
		Eigen::MatrixXd p_com_horizon_;
	};

} // namespace g1_mpc
