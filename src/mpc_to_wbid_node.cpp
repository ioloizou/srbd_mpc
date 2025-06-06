#include <ros/ros.h>
#include <memory>

#include "mpc.cpp"
#include "g1_msgs/State.h"
#include "g1_msgs/SRBD_state.h"
#include "g1_msgs/ContactPoint.h"
#include "std_msgs/Header.h"
#include "rosgraph_msgs/Clock.h"
#include <pal_statistics/pal_statistics_macros.h>

class MPCNode
{
public:
    MPCNode() : nh_(""), received_state_(false), simulation_time_(0.0)
    {
        // Initialize MPC
        mpc_ = std::make_shared<g1_mpc::MPC>();

        // Initialize publishers and subscribers
        sub_current_state_ = nh_.subscribe("/srbd_current", 1, &MPCNode::callbackSrbdCurrent, this);
        pub_mpc_solution_ = nh_.advertise<g1_msgs::SRBD_state>("/mpc_solution", 10);

        // Subscribe to simulation time
        sub_sim_time_ = nh_.subscribe("/simulation_time", 1, &MPCNode::callbackSimTime, this);

        stance_duration_ = 0.25; // seconds
    }

    void callbackSrbdCurrent(const g1_msgs::SRBD_state::ConstPtr &msg)
    {
        // ROS_INFO("Received SRBD current state message");
        Eigen::VectorXd x0(13);
        x0[0] = msg->states_horizon[0].orientation.x;
        x0[1] = msg->states_horizon[0].orientation.y;
        x0[2] = msg->states_horizon[0].orientation.z;
        x0[3] = msg->states_horizon[0].position.x;
        x0[4] = msg->states_horizon[0].position.y;
        x0[5] = msg->states_horizon[0].position.z;
        x0[6] = msg->states_horizon[0].angular_velocity.x;
        x0[7] = msg->states_horizon[0].angular_velocity.y;
        x0[8] = msg->states_horizon[0].angular_velocity.z;
        x0[9] = msg->states_horizon[0].linear_velocity.x;
        x0[10] = msg->states_horizon[0].linear_velocity.y;
        x0[11] = msg->states_horizon[0].linear_velocity.z;
        x0[12] = msg->states_horizon[0].gravity;
        mpc_->setX0(x0);

        // Update current contact points and use for whole horizon
        Eigen::MatrixXd c_horizon(mpc_->horizon_length_, msg->contacts.size() * 3);
        for (int i = 0; i < mpc_->horizon_length_; i++)
        {
            for (size_t j = 0; j < msg->contacts.size(); j++)
            {
                c_horizon(i, j * 3) = msg->contacts[j].position.x;
                c_horizon(i, j * 3 + 1) = msg->contacts[j].position.y;
                c_horizon(i, j * 3 + 2) = msg->contacts[j].position.z;
            }
        }
        mpc_->setCHorizon(c_horizon);

        // Flag that the callback has been received at least once
        received_state_ = true;
    }

    void callbackSimTime(const rosgraph_msgs::Clock &msg)
    {
        simulation_time_ = msg.clock.toSec();
    }

    std::vector<std::vector<int>> gaitPlanner(Eigen::MatrixXd c_horizon, double &landing_position_x_, double &landing_position_y_, bool is_standing = false)
    {
        static double last_switch_time = 0.0;

        // Use simulation time if available, otherwise use ROS time
        double current_time = simulation_time_;

        // Calculate elapsed time since last switch
        double elapsed = current_time - last_switch_time;

        // Gait planning
        const int PLANNING_HORIZON = 20;
        Eigen::MatrixXd contact_planning(PLANNING_HORIZON, mpc_->getNumContacts());
        contact_planning << 1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            0, 0, 1, 1,
                            0, 0, 1, 1,
                            0, 0, 1, 1,
                            0, 0, 1, 1,
                            0, 0, 1, 1,
                            1, 1, 0, 0,
                            1, 1, 0, 0,
                            1, 1, 0, 0,
                            1, 1, 0, 0,
                            1, 1, 0, 0,
                            0, 0, 1, 1,
                            0, 0, 1, 1,
                            0, 0, 1, 1,
                            0, 0, 1, 1,
                            0, 0, 1, 1; 
        
        const int gait_phase = std::floor(current_time / mpc_->getDt()); 
        if (gait_phase >= 10){
            contact_planning << 1, 1, 0, 0,
                                1, 1, 0, 0,
                                1, 1, 0, 0,
                                1, 1, 0, 0,
                                1, 1, 0, 0,
                                0, 0, 1, 1,
                                0, 0, 1, 1,
                                0, 0, 1, 1,
                                0, 0, 1, 1,
                                0, 0, 1, 1,
                                1, 1, 0, 0,
                                1, 1, 0, 0,
                                1, 1, 0, 0,
                                1, 1, 0, 0,
                                1, 1, 0, 0,
                                0, 0, 1, 1,
                                0, 0, 1, 1,
                                0, 0, 1, 1,
                                0, 0, 1, 1,
                                0, 0, 1, 1; 
        }

        // Eigen::MatrixXd contact_planning(PLANNING_HORIZON, mpc_->getNumContacts());
        // contact_planning << 1, 1, 1, 1,
        //                     1, 1, 1, 1,
        //                     1, 1, 1, 1,
        //                     1, 1, 1, 1,
        //                     1, 1, 1, 1,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0,
        //                     0, 0, 1, 1,
        //                     0, 0, 1, 1,
        //                     0, 0, 1, 1,
        //                     0, 0, 1, 1,
        //                     0, 0, 1, 1,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0,
        //                     1, 1, 0, 0;
        
        // const int gait_phase = std::floor(current_time / mpc_->getDt()); 
        // if (gait_phase >= 10){
        //     contact_planning << 0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0,
        //                         0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         0, 0, 1, 1,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0,
        //                         1, 1, 0, 0; 
        // }
        
        const int k = gait_phase % mpc_->horizon_length_;
        // ROS_INFO_STREAM("Gait phase: " << gait_phase << ", k: " << k);
        

        // Create contact horizon based on current phase
        std::vector<std::vector<int>> contact_horizon(mpc_->horizon_length_, std::vector<int>(mpc_->getNumContacts(), 0));
        for (int i = 0; i < mpc_->horizon_length_; i++)
        {
            contact_horizon[i][0] = contact_planning(i + k, 0);
            contact_horizon[i][1] = contact_planning(i + k, 1);
            contact_horizon[i][2] = contact_planning(i + k, 2);
            contact_horizon[i][3] = contact_planning(i + k, 3);
        }
        
        Eigen::MatrixXd contact_horizon_eigen = Eigen::MatrixXd::Zero(contact_horizon.size(), 4);
        for (size_t i = 0; i < contact_horizon.size(); ++i) {
            for (int j = 0; j < 4; ++j) {
                contact_horizon_eigen(i, j) = contact_horizon[i][j];
            }
        }
        // ROS_INFO_STREAM("Contact horizon: \n" << contact_horizon_eigen);
        

        if (is_standing)
        {
            std::fill(contact_horizon.begin(), contact_horizon.end(), std::vector<int>{1, 1, 1, 1});
        }

        /****************************
         * Footstep related         *
         ****************************/
        

        // Need to check when the change happens and the instert raibert heuristic
        double p_swing_foot_land_des_x = 0.0;
        double p_swing_foot_land_des_y = 0.0;

        int sub_phase = k % 5;
        
        // This are offsets for x lower= -0.05 upper= 0.12

        // Because raibert heuristic is only for the com
        double const FOOT_OFFSET_Y_COM = 0.118508;
        double const FOOT_LOWER_X_OFFSET = 0.0293;
        double const FOOT_UPPER_X_OFFSET = 0.1406;

        // Check the sum of all the in row zero colluns if it is 2 otherwise means it is in double stance
        int sum = 0;
        for (int j = 0; j < mpc_->getNumContacts(); ++j) {
            sum += contact_horizon[0][j];
        }

        if (sum == 2) {
            // ROS_INFO_STREAM("Single stance detected, adjusting footstep planning");
            /*
            0, 5 - sub_phase : Current footstep position
            5 - sub_phase, 10 - sub_phase : New footstep position
            10 - sub_phase, 10 : New footstep position
            */
            
            // For the (5 - sub_phase) first nodes we keep current footsteps
            
            RaibertHeuristic(p_swing_foot_land_des_x, 
                             p_swing_foot_land_des_y, 
                             mpc_->getXOpt(), 
                             mpc_->getXRefHor(), 
                             5 - sub_phase, 
                             stance_duration_);

            for (int i = 5 - sub_phase; i < 10 - sub_phase; i++)
            {
                if (contact_horizon[i][0] == 0)
                // If left foot is in swing
                {
                    // Lower left foot
                    c_horizon(i,0) = p_swing_foot_land_des_x - FOOT_LOWER_X_OFFSET;
                    c_horizon(i,1) = p_swing_foot_land_des_y + FOOT_OFFSET_Y_COM;
                    
                    // Upper left foot
                    c_horizon(i,3) = p_swing_foot_land_des_x + FOOT_UPPER_X_OFFSET;
                    c_horizon(i,4) = p_swing_foot_land_des_y + FOOT_OFFSET_Y_COM;
                    
                    // Placeholder for publishing
                    landing_position_x_ = p_swing_foot_land_des_x;
                    landing_position_y_ = p_swing_foot_land_des_y - FOOT_OFFSET_Y_COM; // THIS WORKS WITH OPOSSITE SIGN BUG SOMEWHERE
                }
                else{
                    // Lower right foot
                    c_horizon(i,6) = p_swing_foot_land_des_x - FOOT_LOWER_X_OFFSET;
                    c_horizon(i,7) = p_swing_foot_land_des_y - FOOT_OFFSET_Y_COM;
                    
                    // Upper right foot
                    c_horizon(i,9) = p_swing_foot_land_des_x + FOOT_UPPER_X_OFFSET;
                    c_horizon(i,10) = p_swing_foot_land_des_y - FOOT_OFFSET_Y_COM;

                    // Placeholder for publishing
                    landing_position_x_ = p_swing_foot_land_des_x;
                    landing_position_y_ = p_swing_foot_land_des_y + FOOT_OFFSET_Y_COM;
                }    
            }             
            double p_swing_foot_land_des_x = 0.0;
            double p_swing_foot_land_des_y = 0.0;

            RaibertHeuristic(p_swing_foot_land_des_x, 
                p_swing_foot_land_des_y, 
                mpc_->getXOpt(), 
                mpc_->getXRefHor(), 
                10 - sub_phase, 
                stance_duration_);

            // Now we have a change of switch again
            for (int i = 10 - sub_phase; i < 10; i++){
                if (contact_horizon[i][0] == 0)
                // if left foot is in swing
                {
                    // Lower left foot
                    c_horizon(i,0) = p_swing_foot_land_des_x - FOOT_LOWER_X_OFFSET;
                    c_horizon(i,1) = p_swing_foot_land_des_y + FOOT_OFFSET_Y_COM;
                    
                    // Upper left foot
                    c_horizon(i,3) = p_swing_foot_land_des_x + FOOT_UPPER_X_OFFSET;
                    c_horizon(i,4) = p_swing_foot_land_des_y + FOOT_OFFSET_Y_COM;
                }
                else{
                    // Lower right foot
                    c_horizon(i,6) = p_swing_foot_land_des_x - FOOT_LOWER_X_OFFSET;
                    c_horizon(i,7) = p_swing_foot_land_des_y - FOOT_OFFSET_Y_COM;
                    
                    // Upper right foot
                    c_horizon(i,9) = p_swing_foot_land_des_x + FOOT_UPPER_X_OFFSET;
                    c_horizon(i,10) = p_swing_foot_land_des_y - FOOT_OFFSET_Y_COM;
                }  
            }
            
        }
        else{
            // ROS_INFO_STREAM("Double stance detected in raibert heuristic, no need to adjust footstep planning");    
        }

        mpc_->setCHorizon(c_horizon);
        /****************************
         * End of footstep related  *
         ****************************/

        return contact_horizon;
    }

    void publishMPCSolution(std::vector<std::vector<int>> contact_horizon, double &landing_position_x_, double &landing_position_y_)
    {
        g1_msgs::SRBD_state srbd_state_msg;
        srbd_state_msg.header.stamp = ros::Time(simulation_time_);
        srbd_state_msg.header.frame_id = "SRBD";

        // Get the optimal state trajectory and control inputs
        const Eigen::MatrixXd &x_opt = mpc_->getXOpt();
        const Eigen::MatrixXd &u_opt = mpc_->getUOpt();

        for (int i = 0; i<mpc_->horizon_length_; i++)
        {
            g1_msgs::State state_msg;
            state_msg.trajectory_index = i;
            state_msg.orientation.x = x_opt(i, 0);
            state_msg.orientation.y = x_opt(i, 1);
            state_msg.orientation.z = x_opt(i, 2);
            state_msg.position.x = x_opt(i, 3);
            state_msg.position.y = x_opt(i, 4);
            state_msg.position.z = x_opt(i, 5);
            state_msg.angular_velocity.x = x_opt(i, 6);
            state_msg.angular_velocity.y = x_opt(i, 7);
            state_msg.angular_velocity.z = x_opt(i, 8);
            state_msg.linear_velocity.x = x_opt(i, 9);
            state_msg.linear_velocity.y = x_opt(i, 10);
            state_msg.linear_velocity.z = x_opt(i, 11);
            state_msg.gravity = x_opt(i, 12);
            
            srbd_state_msg.states_horizon.push_back(state_msg);
        }

        // Set contact forces from optimized controls
        std::vector<std::string> contact_names = {
            "left_foot_line_contact_lower",
            "left_foot_line_contact_upper",
            "right_foot_line_contact_lower",
            "right_foot_line_contact_upper"};
        
        Eigen::MatrixXd c_horizon = mpc_->getCHorizon();
        for (int i = 0; i < 4; i++)
        {
            g1_msgs::ContactPoint contact_point_msg;
            contact_point_msg.name = contact_names[i];
            contact_point_msg.force.x = u_opt(0, i * 3);
            contact_point_msg.force.y = u_opt(0, i * 3 + 1);
            contact_point_msg.force.z = u_opt(0, i * 3 + 2);
            contact_point_msg.active = contact_horizon[0][i]; // Use first step of horizon
            contact_point_msg.position.x = c_horizon(0, i * 3);
            contact_point_msg.position.y = c_horizon(0, i * 3 + 1);
            contact_point_msg.position.z = c_horizon(0, i * 3 + 2);
            srbd_state_msg.contacts.push_back(contact_point_msg);
        }

        srbd_state_msg.landing_position.x = landing_position_x_; 
        srbd_state_msg.landing_position.y = landing_position_y_;
        // Not needed z since it will be calculated from the trajectory 
        srbd_state_msg.landing_position.z = 0.0;

        // Publish the message
        pub_mpc_solution_.publish(srbd_state_msg);
    }

    void update()
    {
        // Ensure we have received an initial state before proceeding
        if (!received_state_)
        {
            // ROS_WARN("No state callback received yet; skipping MPC update.");
            return;
        }
        // ROS_INFO("Updating MPC...");
        // Get contact horizon (which feet are in contact)

        
        // Get current time
        double time = simulation_time_;
        
        bool is_walking_forward = true;
        
        Eigen::MatrixXd x_ref_hor = mpc_->getXRefHor();
        
        // Commanded Velocity (roll, pitch, yaw, x, y, z)
        Eigen::VectorXd v_cmd(6);
        v_cmd << 0.0, 0.0, 0.0, 0.35, 0.0, 0.0;

        // Reference trajectory is a simple euler integration of velocity command
        if (is_walking_forward)
        {
            for (int i = 0; i < mpc_->horizon_length_; i++)
            {
                x_ref_hor(i, 4) = 7.44339342e-05;
                x_ref_hor(i, 5) = 5.97983255e-01;
                // The velocity is in world frame 
                // X velocity integration
                x_ref_hor(i, 3) = mpc_->getX0()(3) + v_cmd(3) * i * mpc_->getDt();
                
                // Y velocity integration
                // x_ref_hor(i, 4) = mpc_->getX0()(4) + v_cmd(4) * i * mpc_->getDt();

                // Set the x, y reference velocity
                x_ref_hor(i, 9) = v_cmd(3);
                // x_ref_hor(i, 10) = v_cmd(4);

                // Angular velocity integration (only yaw)
                // In world frame
                x_ref_hor(i, 2) = 0; 
                // mpc_->getX0()(2) + v_cmd(2) * i * mpc_->getDt();
                
                // Set the angular velocity reference
                x_ref_hor(i, 8) = v_cmd(2) ;

                x_ref_hor(i, 12) = mpc_->g_;
            }
        }

        else{
            // Update reference trajectory
            x_ref_hor.block(0, 3, 1, 3) << 5.26790425e-02, 7.44339342e-05, 5.97983255e-01;
            
            double radius = 0.05;
            double x_const = x_ref_hor(0, 3);
            double y_center = x_ref_hor(0, 4);
            double z_center = x_ref_hor(0, 5);
            
            double speed = 0.5;
            for (int i = 0; i < mpc_->horizon_length_; i++)
            {
                x_ref_hor(i, 3) = x_const;
                x_ref_hor(i, 4) = y_center;
                // - radius/16 * std::cos(speed*M_PI*time);
                x_ref_hor(i, 5) = z_center;
                // - radius/2 * std::cos(speed*5*M_PI*time);
                x_ref_hor(i, 12) = mpc_->g_;
            }
        }
        mpc_->setXRefHor(x_ref_hor);
        
        // This should be done inside the MPC class
        // Extract COM positions for the horizon
        Eigen::MatrixXd p_com_horizon_matrix(mpc_->horizon_length_, 3);
        for (int i = 0; i < mpc_->horizon_length_; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                p_com_horizon_matrix(i, j) = x_ref_hor(i, j + 3);
            }
        }
        
        double landing_position_x_ = 0.0;
        double landing_position_y_ = 0.0;
        
        std::vector<std::vector<int>> contact_horizon = gaitPlanner(mpc_->getCHorizon(), landing_position_x_, landing_position_y_);
        
        // Update MPC solution
        auto start_time = ros::WallTime::now();

        Eigen::VectorXd x_current = mpc_->getX0();

        // Convert std::vector<std::vector<int>> to Eigen::MatrixXd
        Eigen::MatrixXd contact_horizon_matrix(contact_horizon.size(), contact_horizon[0].size());
        for (size_t i = 0; i < contact_horizon.size(); ++i)
        {
            for (size_t j = 0; j < contact_horizon[i].size(); ++j)
            {
                contact_horizon_matrix(i, j) = contact_horizon[i][j];
            }
        }

        Eigen::MatrixXd c_horizon = mpc_->getCHorizon();
        // mpc_->debugPrintMatrixDimensions();

        // Pass matrices instead of vectors
        mpc_->updateMPC(contact_horizon_matrix, c_horizon, p_com_horizon_matrix, &x_current, false);

        mpc_solve_time_ = (ros::WallTime::now() - start_time).toSec(); // seconds

        pal_statistics::RegistrationsRAII registrations;
        REGISTER_VARIABLE("/mpc_statistics", "MPC Solve time", &mpc_solve_time_, &registrations);
        PUBLISH_STATISTICS("/mpc_statistics");

        // debugMPCVariables(contact_horizon_matrix, c_horizon, p_com_horizon_matrix);
        // Publish solution
        publishMPCSolution(contact_horizon, landing_position_x_, landing_position_y_);
    }

    void RaibertHeuristic(double &p_swing_foot_land_des_x, double &p_swing_foot_land_des_y, Eigen::MatrixXd const &x_opt, Eigen::MatrixXd const &x_ref_hor, int const i, double const stance_duration){
        Eigen::Vector3d p_com = x_opt.row(i).segment(3, 3);
        double p_com_x = p_com(0); 
        double p_com_y = p_com(1);
        
        Eigen::Vector3d p_dot_com = x_opt.row(i).segment(9, 3);
        double p_dot_com_x = p_dot_com(0);
        double p_dot_com_y = p_dot_com(1);
        
        Eigen::Vector3d p_dot_com_des = x_ref_hor.row(i).segment(9, 3);
        double p_dot_com_des_x = p_dot_com_des(0);
        double p_dot_com_des_y = p_dot_com_des(1); 
        
        const double k_vel_gain = 0.02;

        // TODO add the offset from the CoM to the foot
        p_swing_foot_land_des_x = p_com_x + p_dot_com_x * stance_duration * 0.5 * mpc_->getDt() + k_vel_gain * (p_dot_com_x - p_dot_com_des_x);
        p_swing_foot_land_des_y = p_com_y + p_dot_com_y * stance_duration * 0.5 * mpc_->getDt() + k_vel_gain * (p_dot_com_y - p_dot_com_des_y);

        // Centrifugal term can be added later
    }

    // Debug function to print all MPC-related variables
    void debugMPCVariables(Eigen::MatrixXd const &contact_horizon_matrix,
                           Eigen::MatrixXd const &c_horizon_matrix,
                           Eigen::MatrixXd const &p_com_horizon_matrix)
    {
        // ROS_INFO_STREAM("MPC x0: \n"
        //                 << mpc_->getX0());
        // ROS_INFO_STREAM("MPC x_ref_hor: \n"
        //                 << mpc_->getXRefHor());
        // ROS_INFO_STREAM("c_horizon_matrix: \n"
        //                 << c_horizon_matrix);
        ROS_INFO_STREAM("MPC c_horizon: \n"
                        << mpc_->getCHorizon());
        // ROS_INFO_STREAM("p_com_horizon_matrix: \n"
        //                 << p_com_horizon_matrix);
        // ROS_INFO_STREAM("MPC p_com_horizon: \n"
        //                 << mpc_->getPComHorizon());
        ROS_INFO_STREAM("contact_horizon_matrix: \n"
                        << contact_horizon_matrix);
        // ROS_INFO_STREAM("MPC contact_horizon: \n"
                        // << mpc_->getContactHorizon());
        // ROS_INFO_STREAM("MPC x_opt: \n"
        //                 << mpc_->getXOpt());
        // ROS_INFO_STREAM("MPC u_opt: \n"
        //                 << mpc_->getUOpt());
        // ROS_INFO_STREAM("MPC contact forces (first step): \n"
        //                 << mpc_->getUOpt().row(0));
        // ROS_INFO_STREAM("MPC solve time: " << mpc_solve_time_ << " ms");
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_current_state_;
    ros::Subscriber sub_sim_time_;
    ros::Publisher pub_mpc_solution_;

    std::shared_ptr<g1_mpc::MPC> mpc_;
    bool received_state_;    // Flag to indicate state received
    double simulation_time_; // Store simulation time
    double mpc_solve_time_;  // For statistics
    double stance_duration_; // Duration of stance phase
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mpc_to_wbid");

    MPCNode node;
    ros::Rate rate(400);

    while (ros::ok())
    {
        ros::spinOnce();
        node.update();
        rate.sleep();
    }

    return 0;
}
