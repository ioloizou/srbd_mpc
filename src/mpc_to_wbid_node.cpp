#include <ros/ros.h>
#include <memory>

#include "mpc.cpp"
#include "g1_msgs/SRBD_state.h"
#include "g1_msgs/ContactPoint.h"
#include "std_msgs/Header.h"
#include <pal_statistics/pal_statistics_macros.h>

class MPCNode
{
public:
    MPCNode() : nh_(""), received_state_(false)
    {
        // Initialize MPC
        mpc_ = std::make_shared<g1_mpc::MPC>();

        // Initialize publishers and subscribers
        sub_current_state_ = nh_.subscribe("/srbd_current", 1, &MPCNode::callbackSrbdCurrent, this);
        pub_mpc_solution_ = nh_.advertise<g1_msgs::SRBD_state>("/mpc_solution", 10);


        // ROS_INFO("MPC node initialized successfully");
    }

    void callbackSrbdCurrent(const g1_msgs::SRBD_state::ConstPtr &msg)
    {
        // ROS_INFO("Received SRBD current state message");
        Eigen::VectorXd x0(13);
        x0[0] = msg->orientation.x;
        x0[1] = msg->orientation.y;
        x0[2] = msg->orientation.z;
        x0[3] = msg->position.x;
        x0[4] = msg->position.y;
        x0[5] = msg->position.z;
        x0[6] = msg->angular_velocity.x;
        x0[7] = msg->angular_velocity.y;
        x0[8] = msg->angular_velocity.z;
        x0[9] = msg->linear_velocity.x;
        x0[10] = msg->linear_velocity.y;
        x0[11] = msg->linear_velocity.z;
        x0[12] = msg->gravity;
        mpc_->setX0(x0);

        // Update contact points
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

    std::vector<std::vector<int>> gaitPlanner(bool is_standing = false)
    {
        static ros::Time last_switch_time = ros::Time::now();
        static int current_phase = 0;
        const double stance_duration = 0.4; // seconds

        // Calculate elapsed time since last switch
        ros::Time current_time = ros::Time::now();
        double elapsed = (current_time - last_switch_time).toSec();

        // Check if it's time to switch gait pattern
        if (elapsed >= stance_duration)
        {
            current_phase = (current_phase + 1) % 2;
            last_switch_time = current_time;
        }

        // Create contact horizon based on current phase
        std::vector<std::vector<int>> contact_horizon;
        for (int i = 0; i < mpc_->horizon_length_; i++)
        {
            if (current_phase == 0)
            {
                contact_horizon.push_back({1, 1, 0, 0});
            }
            else
            {
                contact_horizon.push_back({0, 0, 1, 1});
            }
        }

        if (is_standing)
        {
            std::fill(contact_horizon.begin(), contact_horizon.end(), std::vector<int>{1, 1, 1, 1});
        }

        return contact_horizon;
    }

    void publishMPCSolution(std::vector<std::vector<int>> contact_horizon)
    {
        g1_msgs::SRBD_state srbd_state_msg;
        srbd_state_msg.header.stamp = ros::Time::now();
        srbd_state_msg.header.frame_id = "SRBD";

        // Get the optimal state trajectory and control inputs
        const Eigen::MatrixXd &x_opt = mpc_->getXOpt();
        const Eigen::MatrixXd &u_opt = mpc_->getUOpt();

        // Set orientation and position from optimized state
        srbd_state_msg.orientation.x = x_opt(1, 0);
        srbd_state_msg.orientation.y = x_opt(1, 1);
        srbd_state_msg.orientation.z = x_opt(1, 2);
        srbd_state_msg.position.x = x_opt(1, 3);
        srbd_state_msg.position.y = x_opt(1, 4);
        srbd_state_msg.position.z = x_opt(1, 5);
        srbd_state_msg.angular_velocity.x = x_opt(1, 6);
        srbd_state_msg.angular_velocity.y = x_opt(1, 7);
        srbd_state_msg.angular_velocity.z = x_opt(1, 8);
        srbd_state_msg.linear_velocity.x = x_opt(1, 9);
        srbd_state_msg.linear_velocity.y = x_opt(1, 10);
        srbd_state_msg.linear_velocity.z = x_opt(1, 11);
        srbd_state_msg.gravity = x_opt(1, 12);

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
        if (!received_state_) {
            // ROS_WARN("No state callback received yet; skipping MPC update.");
            return;
        }
        // ROS_INFO("Updating MPC...");
        // Get contact horizon (which feet are in contact)

        // If using gait patterns:
        std::vector<std::vector<int>> contact_horizon = gaitPlanner();

        // Get current time
        double time = ros::Time::now().toSec();

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
            x_ref_hor(i, 5) = z_center; 
            // - radius/2 * std::cos(speed*M_PI*time);
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
        mpc_->updateMPC(contact_horizon_matrix, c_horizon, p_com_horizon_matrix, &x_current, true);
        
        mpc_solve_time_ = (ros::WallTime::now() - start_time).toSec() * 1000.0; // ms
        
        pal_statistics::RegistrationsRAII registrations;
        REGISTER_VARIABLE("/mpc_statistics", "MPC Solve time", &mpc_solve_time_, &registrations);
        PUBLISH_STATISTICS("/mpc_statistics");

        
        // debugMPCVariables(contact_horizon_matrix, c_horizon, p_com_horizon_matrix);
        // Publish solution
        publishMPCSolution(contact_horizon);
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
    ros::Publisher pub_mpc_solution_;

    std::shared_ptr<g1_mpc::MPC> mpc_;
    bool received_state_; // New flag to indicate state received
    double mpc_solve_time_; // For statistics
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mpc_to_wbid");

    MPCNode node;
    ros::Rate rate(300);

    while (ros::ok())
    {
        ros::spinOnce();
        node.update();
        rate.sleep();
    }

    return 0;
}
