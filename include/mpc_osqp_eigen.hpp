#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>
#include <iostream>
#include <chrono>

// I will refactor eveything to use the same naming convention as my python once it works
const double g = -9.81;
const int LEGS = 4;
const int NUM_STATE = 13;
const int NUM_DOF = 3 * LEGS;
const int HORIZON_LENGTH = 15;
const double dt = 0.04;
const int NUM_BOUNDS = 5;

const Eigen::Matrix3d INERTIA_BODY = (Eigen::Matrix3d() << 3.20564, 4.35027e-05, 0.425526,
                                                            4.35027e-05, 3.05015, -0.00065082,
                                                            0.425526, -0.00065082, 0.552353).finished();

const double ROBOT_MASS = 34.13385728;

class MPC{
public:
    
    MPC(){
        mu = 0.3;
        fz_min = 10;
        fz_max = 666;        
    }
    void initMatricesZero(){
        A_matrix_continuous = Eigen::Matrix<double, NUM_STATE, NUM_STATE>::Zero();
        A_matrix_discrete = Eigen::Matrix<double, NUM_STATE, NUM_STATE>::Zero();
        B_matrix_continuous = Eigen::Matrix<double, NUM_STATE, NUM_DOF>::Zero();
        B_matrix_continuous_list = Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_DOF>::Zero();
        B_matrix_discrete_list = Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_DOF>::Zero();
        B_matrix_discrete = Eigen::Matrix<double, NUM_STATE, NUM_DOF>::Zero();
        Aqp_matrix = Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_STATE>::Zero();
        Bqp_matrix = Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_DOF * HORIZON_LENGTH>::Zero();
    }
    double extractPsi(Eigen::MatrixXd ref_body_plan){
        double yaw_sum = 0;
        for (int i = 0; i < HORIZON_LENGTH; i++)
        {
            yaw_sum += ref_body_plan(i, 5);
        }
        double psi = yaw_sum / HORIZON_LENGTH;
        return psi;
    }
    void vectorToSkewSymmetric(Eigen::Vector3d vector, Eigen::Matrix3d &skew_symmetric){
        skew_symmetric << 0, -vector(2), vector(1),
                          vector(2), 0, -vector(0),
                          -vector(1), vector(0), 0;
    }

    Eigen::Matrix3d setRotationMatrixT(double average_yaw){
        double psi = average_yaw;
        Rotation_z << cos(psi), sin(psi), 0,
                     -sin(psi), cos(psi), 0,
                             0,        0, 1;
        return Rotation_z;
    }

    void setQMatrix(){

        Q_matrix = Eigen::SparseMatrix<double>(NUM_STATE * HORIZON_LENGTH, NUM_STATE * HORIZON_LENGTH);
        for (int i = 0; i < NUM_STATE * HORIZON_LENGTH; i++)
        {
            q_weights << 7e5, 7e4, 1e4,
                         5e5, 5e5, 3e6,
                         3e3, 3e3, 3e3,
                         5e3, 1e3, 1e4, 0;
            Q_matrix.insert(i, i) = 2 * q_weights(i % NUM_STATE);
        }
    }

    void setRMatrix(){
        R_matrix = Eigen::SparseMatrix<double>(NUM_DOF * HORIZON_LENGTH, NUM_DOF * HORIZON_LENGTH);
        for (int i = 0; i < NUM_DOF * HORIZON_LENGTH; i++)
        {
            R_matrix.insert(i, i) = 2 * r_weights(i % NUM_DOF);
        }
    }

    void setAMatrixContinuous(Eigen::Matrix3d Rotation_z){
        A_matrix_continuous.block<3, 3>(0, 6) = Rotation_z;
        A_matrix_continuous.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity();
        A_matrix_continuous(11, 12) = 1; 

    }

    void setAMatrixDiscrete(double &first_element_duration){
        Eigen::MatrixXd A_matrix_continuous_sampled = A_matrix_continuous*dt;
        A_matrix_continuous_sampled.row(0) = A_matrix_continuous.row(0)*first_element_duration;
        A_matrix_discrete = Eigen::Matrix<double, NUM_STATE, NUM_STATE>::Identity(NUM_STATE, NUM_STATE) + A_matrix_continuous_sampled;        
    }

    void setBMatrixContinuous(Eigen::MatrixXd foot_positions, Eigen::Matrix3d Rotation_z){
       
        Eigen::Matrix3d A1_INERTIA_WORLD;
        A1_INERTIA_WORLD = Rotation_z * INERTIA_BODY * Rotation_z.transpose();
        
        for (int i = 0; i < HORIZON_LENGTH; i++)
        {                
            for (int j=0; j<LEGS; j++)
            {
                Eigen::Vector3d r_leg = foot_positions.row(i).segment(3*j, 3);
                Eigen::Matrix3d r_leg_skew;
                vectorToSkewSymmetric(r_leg, r_leg_skew);

                B_matrix_continuous.block<3, 3>(6, 3*j) = A1_INERTIA_WORLD.inverse() * Rotation_z*r_leg_skew;
                B_matrix_continuous.block<3, 3>(9, 3*j) = Eigen::Matrix3d::Identity() * (1/ROBOT_MASS);

            }
            B_matrix_continuous_list.block<NUM_STATE, NUM_DOF>(i*NUM_STATE, 0) = B_matrix_continuous;
            
        }
    }

    void setBMatrixDiscrete(double &first_element_duration){
        Eigen::Matrix<double, NUM_STATE, NUM_DOF> current_B;
        for (int i = 0; i < HORIZON_LENGTH; i++)
        {
            current_B = B_matrix_continuous_list.block<NUM_STATE, NUM_DOF>(i*NUM_STATE,0);
            B_matrix_discrete = current_B * dt;
            B_matrix_discrete.row(0) = current_B.row(0)*first_element_duration;
            B_matrix_discrete_list.block<NUM_STATE, NUM_DOF>(i*NUM_STATE, 0) = B_matrix_discrete;
        }        
    }

    void setAqpMatrix(){
        for (int i = 0; i < HORIZON_LENGTH; i++)
        {
            if (i == 0)
            {  
                Aqp_matrix.block<NUM_STATE, NUM_STATE>(i * NUM_STATE, 0) = A_matrix_discrete;
            }
            else   
            {
                Aqp_matrix.block<NUM_STATE, NUM_STATE>(i * NUM_STATE, 0) = Aqp_matrix.block<NUM_STATE, NUM_STATE>((i - 1) * NUM_STATE, 0) * A_matrix_discrete;
            }
        }
    }

    void setBqpMatrix(){
        for (int i = 0; i< HORIZON_LENGTH; i++)
        {
            for (int j=0; j<= i; j++){
                if (i - j== 0)
                {
                    Bqp_matrix.block<NUM_STATE, NUM_DOF>(i * NUM_STATE, j*NUM_DOF) = B_matrix_discrete_list.block<NUM_STATE, NUM_DOF>(j*NUM_STATE, 0);
                }
                else
                {   
                    Bqp_matrix.block<NUM_STATE, NUM_DOF>(i * NUM_STATE, j * NUM_DOF) = 
                    Aqp_matrix.block<NUM_STATE, NUM_STATE>((i -j - 1) * NUM_STATE, 0) * B_matrix_discrete_list.block<NUM_STATE, NUM_DOF>(j*NUM_STATE,0);
                }
            }           
        }
    }

    void setAcMatrix(){
        g_block.resize(NUM_BOUNDS , NUM_DOF/LEGS);
        g_block << 1,0,mu,  
                   1,0,-mu, 
                   0,1,mu,  
                   0,1,-mu, 
                   0,0,1;   

        Ac_matrix = Eigen::SparseMatrix<double>(NUM_BOUNDS * LEGS * HORIZON_LENGTH, NUM_DOF * HORIZON_LENGTH);
        std::vector<Eigen::Triplet<double>> tripletList;
        
        tripletList.reserve(9 * 40);  

        int row_offset = 0;
        int col_offset = 0;

        while (row_offset + g_block.rows() <= Ac_matrix.rows() && col_offset + g_block.cols() <= Ac_matrix.cols())
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

        Ac_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
        Ac_matrix.makeCompressed();
    }

    void setBounds(std::vector<std::vector<bool>> contact){
        
        Eigen::VectorXd lower_bounds(NUM_BOUNDS * LEGS);
        Eigen::VectorXd upper_bounds(NUM_BOUNDS * LEGS);
        
        lower_bounds_horizon = Eigen::VectorXd::Zero(NUM_BOUNDS * LEGS * HORIZON_LENGTH);
        upper_bounds_horizon = Eigen::VectorXd::Zero(NUM_BOUNDS * LEGS * HORIZON_LENGTH);

        int horizon_step = 0;
        
        while (horizon_step < HORIZON_LENGTH)
        {
            for (int i=0; i<LEGS; i++)
            {
            lower_bounds.segment(i*NUM_BOUNDS, 5) << 0,                                        
                                                    -std::numeric_limits<double>::infinity(),  
                                                     0,                                        
                                                    -std::numeric_limits<double>::infinity(),  
                                                    fz_min*contact[horizon_step][i];                   
                                                     
            upper_bounds.segment(i*NUM_BOUNDS, 5) << std::numeric_limits<double>::infinity(),  
                                                     0,                                        
                                                     std::numeric_limits<double>::infinity(),  
                                                     0,                                        
                                                     fz_max*contact[horizon_step][i];                   
            }

            lower_bounds_horizon.segment(horizon_step*NUM_BOUNDS*LEGS, NUM_BOUNDS * LEGS) = lower_bounds;
            upper_bounds_horizon.segment(horizon_step*NUM_BOUNDS*LEGS, NUM_BOUNDS * LEGS) = upper_bounds;
            horizon_step += 1;
        }
        
    }

    void setHessian(){   
        hessian = Bqp_matrix.transpose() * Q_matrix * Bqp_matrix + R_matrix;
    }

    void setGradient(Eigen::MatrixXd grf_plan_, Eigen::VectorXd current_state_, Eigen::MatrixXd ref_body_plan_){
        
        gradient.resize(NUM_DOF * HORIZON_LENGTH, 1);

        Eigen::VectorXd states_ref_stacked = Eigen::VectorXd::Zero(NUM_STATE * HORIZON_LENGTH);
        for (int i = 0; i < HORIZON_LENGTH; i++){
            states_ref_stacked.segment(i*NUM_STATE, NUM_STATE) = ref_body_plan_.row(i).transpose();
        }

        gradient = Bqp_matrix.transpose() * Q_matrix * ((Aqp_matrix *current_state_) - states_ref_stacked);
    }

    void computeRollout(Eigen::VectorXd &u, Eigen::MatrixXd &x, Eigen::VectorXd current_state)
    {
        x.row(0) = current_state.transpose();

        Eigen::VectorXd x_next = current_state;
        for (int i = 1; i < HORIZON_LENGTH; i++)
        {
            x_next = A_matrix_discrete * x_next + B_matrix_discrete_list.block<NUM_STATE, NUM_DOF>((i-1)*NUM_STATE, 0) * u.segment((i-1) * NUM_DOF, NUM_DOF);            
            x.row(i) = x_next.transpose();
        }
    }

    void computeRollout(Eigen::MatrixXd &u, Eigen::MatrixXd &x, Eigen::VectorXd current_state)
    {
        x.conservativeResize(HORIZON_LENGTH, NUM_STATE);

        x.row(0) = current_state.transpose();

        Eigen::VectorXd x_next = current_state;
        for (int i = 1; i < HORIZON_LENGTH; i++)
        {          
            x_next = A_matrix_discrete * x_next + B_matrix_discrete_list.block<NUM_STATE, NUM_DOF>((i-1)*NUM_STATE, 0) * u.row(i-1).transpose();
            x.row(i) = x_next.transpose();
        }
    }


    Eigen::VectorXd solveQP(){
        //Instantiate the solver
        OsqpEigen::Solver solver;

        auto t0 = std::chrono::high_resolution_clock::now();
        //Configure the solver
        if (!solver.isInitialized()){
            solver.settings()->setVerbosity(false);
            solver.settings()->setWarmStart(true);
            solver.data()->setNumberOfVariables(NUM_DOF*HORIZON_LENGTH);
            solver.data()->setNumberOfConstraints(NUM_BOUNDS*LEGS*HORIZON_LENGTH);
            solver.data()->setLinearConstraintsMatrix(Ac_matrix);
            solver.data()->setHessianMatrix(hessian);
            solver.data()->setGradient(gradient);
            solver.data()->setLowerBound(lower_bounds_horizon);
            solver.data()->setUpperBound(upper_bounds_horizon);
            solver.initSolver();
            }
        else{
            solver.updateGradient(gradient);
            solver.updateHessianMatrix(hessian);
            solver.updateBounds(lower_bounds_horizon, upper_bounds_horizon);
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
 
        return result;
    }
    // Parameters
    double mu;
    double fz_min;
    double fz_max;
    
    //Matrices declaration
    Eigen::Matrix<double, NUM_STATE, NUM_STATE> A_matrix_continuous;
    Eigen::Matrix<double, NUM_STATE, NUM_STATE> A_matrix_discrete;
    Eigen::Matrix<double, NUM_STATE, NUM_DOF> B_matrix_continuous;
    Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_DOF> B_matrix_continuous_list;
    Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_DOF> B_matrix_discrete_list;
    Eigen::Matrix<double, NUM_STATE, NUM_DOF> B_matrix_discrete; 
    Eigen::SparseMatrix<double> Q_matrix;
    Eigen::SparseMatrix<double> R_matrix;
    Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_STATE> Aqp_matrix;
    Eigen::Matrix<double, NUM_STATE * HORIZON_LENGTH, NUM_DOF * HORIZON_LENGTH> Bqp_matrix;

    Eigen::Matrix<double,NUM_BOUNDS , NUM_DOF/LEGS> g_block;
    Eigen::SparseMatrix<double> Ac_matrix;
    
    Eigen::VectorXd lower_bounds_horizon;
    Eigen::VectorXd upper_bounds_horizon;
    
    Eigen::Matrix<double, NUM_DOF * HORIZON_LENGTH, 1> gradient;
    Eigen::SparseMatrix<double> hessian;

    Eigen::Matrix<double, 3, 3> Rotation_z;

    bool is_first_run = true;  //to be set to false after first iteration, so that the initial guess is correctly set to hot-start the solver

    Eigen::VectorXd q_weights = Eigen::VectorXd(NUM_STATE);
    Eigen::VectorXd r_weights = Eigen::VectorXd::Ones(NUM_DOF)*0.001;
};

