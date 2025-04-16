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

    std::vector<std::vector<int>> gaitPlanner(bool is_standing = false, Eigen::MatrixXd &c_horizon)
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
        
        const int k = gait_phase % mpc_->horizon_length_;
        ROS_INFO_STREAM("Gait phase: " << gait_phase << ", k: " << k);
        

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
        ROS_INFO_STREAM("Contact horizon: \n" << contact_horizon_eigen);
        

        if (is_standing)
        {
            std::fill(contact_horizon.begin(), contact_horizon.end(), std::vector<int>{1, 1, 1, 1});
        }

        /****************************
         * Footstep related         *
         ****************************/
        // To get the current and then modify with raibert
        c_horizon = mpc_->getCHorizon();

        // Need to check when the change happens and the instert raibert heuristic
        double p_swing_foot_land_des_x = 0.0;
        double p_swing_foot_land_des_y = 0.0;

        int remaining_stance_steps = k % 5; 

        


        return contact_horizon;
    }

    Eigen::MatrixXd FootstepPlanner(){

        auto current_time = simulation_time_;

        int gait_phase = std::floor(current_time / mpc_->getDt()); 
        int k = gait_phase % mpc_->horizon_length_;

        

        // To get the current and then modify with raibert
        Eigen::MatrixXd c_horizon = mpc_->getCHorizon();

        // Need to check when the change happens and the instert raibert heuristic
        double p_swing_foot_land_des_x = 0.0;
        double p_swing_foot_land_des_y = 0.0;
        
        int i=0;
        int remaining_stance_steps = k % 5; 

        // When i=0 is the current k
        if (k == 0){
            RaibertHeuristic(p_swing_foot_land_des_x, p_swing_foot_land_des_y, mpc_->getXOpt(), mpc_->getXRefHor(), i, stance_duration_, k);  
            for (int i = 0; i < 5; i++){
                c_horizon(i, 0) = p_swing_foot_land_des_x;
                c_horizon(i, 1) = p_swing_foot_land_des_y;
            }
        }


        /*
        Get the current footstep position.
        do
         Check if have swing. (swing when k == 0)
         If yes, then modify the footstep position.
        while (i < mpc_->horizon_length_)
        */


        // Update the contact horizon with the new footstep
        // for (int i = 0; i < mpc_->horizon_length_; i++)
        // {
        //     for (size_t j = 0; j < 3; j++)
        //     {
        //         c_horizon(i, j * 3) = msg->contacts[j].position.x;
        //         c_horizon(i, j * 3 + 1) = msg->contacts[j].position.y;
        //         c_horizon(i, j * 3 + 2) = msg->contacts[j].position.z;
        //     }
        // }

        return c_horizon;
    } 

    void publishMPCSolution(std::vector<std::vector<int>> contact_horizon)
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

        for (int i = 0; i < 4; i++)
        {
            g1_msgs::ContactPoint contact_point_msg;
            contact_point_msg.name = contact_names[i];
            contact_point_msg.force.x = u_opt(0, i * 3);
            contact_point_msg.force.y = u_opt(0, i * 3 + 1);
            contact_point_msg.force.z = u_opt(0, i * 3 + 2);
            contact_point_msg.active = contact_horizon[0][i]; // Use first step of horizon
            srbd_state_msg.contacts.push_back(contact_point_msg);
        }

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

        // If using gait patterns:
        std::vector<std::vector<int>> contact_horizon = gaitPlanner();

        // Get current time
        double time = simulation_time_;

        // Update reference trajectory
        Eigen::MatrixXd x_ref_hor = mpc_->getXRefHor();
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
        publishMPCSolution(contact_horizon);
    }

    void RaibertHeuristic(double p_swing_foot_land_des_x, double p_swing_foot_land_des_y, Eigen::MatrixXd const &x_opt, Eigen::MatrixXd const &x_ref_hor, int const i, double const stance_duration, double k){
        Eigen::Vector3d p_com = x_opt.row(i).segment(3, 3);
        double p_com_x = p_com(0); 
        double p_com_y = p_com(1);
        
        Eigen::Vector3d p_dot_com = x_opt.row(i).segment(9, 3);
        double p_dot_com_x = p_dot_com(0);
        double p_dot_com_y = p_dot_com(1);
        
        Eigen::Vector3d p_dot_com_des = x_ref_hor.row(i).segment(9, 3);
        double p_dot_com_des_x = p_dot_com_des(0);
        double p_dot_com_des_y = p_dot_com_des(1); 
        
        k = stance_duration * 0.5;

        p_swing_foot_land_des_x = p_com_x + p_dot_com_x * stance_duration * 0.5 + k*(p_dot_com_x - p_dot_com_des_x);
        p_swing_foot_land_des_y = p_com_y + p_dot_com_y * stance_duration * 0.5 + k*(p_dot_com_y - p_dot_com_des_y);

        // Centrifugal term can be added later
    }

    // Debug function to print all MPC-related variables
    void debugMPCVariables(Eigen::MatrixXd const &contact_horizon_matrix,
                           Eigen::MatrixXd const &c_horizon_matrix,
                           Eigen::MatrixXd const &p_com_horizon_matrix)
    {
        ROS_INFO_STREAM("MPC x0: \n"
                        << mpc_->getX0());
        ROS_INFO_STREAM("MPC x_ref_hor: \n"
                        << mpc_->getXRefHor());
        ROS_INFO_STREAM("c_horizon_matrix: \n"
                        << c_horizon_matrix);
        ROS_INFO_STREAM("MPC c_horizon: \n"
                        << mpc_->getCHorizon());
        ROS_INFO_STREAM("p_com_horizon_matrix: \n"
                        << p_com_horizon_matrix);
        ROS_INFO_STREAM("MPC p_com_horizon: \n"
                        << mpc_->getPComHorizon());
        ROS_INFO_STREAM("contact_horizon_matrix: \n"
                        << contact_horizon_matrix);
        ROS_INFO_STREAM("MPC contact_horizon: \n"
                        << mpc_->getContactHorizon());
        ROS_INFO_STREAM("MPC x_opt: \n"
                        << mpc_->getXOpt());
        ROS_INFO_STREAM("MPC u_opt: \n"
                        << mpc_->getUOpt());
        ROS_INFO_STREAM("MPC contact forces (first step): \n"
                        << mpc_->getUOpt().row(0));
        ROS_INFO_STREAM("MPC solve time: " << mpc_solve_time_ << " ms");
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
