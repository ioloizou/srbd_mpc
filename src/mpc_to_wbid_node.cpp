#include <ros/ros.h>
#include <memory>
#include "mpc_osqp_eigen_header.hpp"
#include "g1_msgs/SRBD_state.h"
#include "g1_msgs/ContactPoint.h"
#include "std_msgs/Header.h"
#include <pal_statistics/pal_statistics.h>

// static int hi_counter = 0;

class MPCNode
{
public:
  MPCNode() : nh_("")
  {
    // Initialize MPC
    mpc_ = std::make_shared<g1_mpc::MPC>();
    mpc_->initMatrices();

    // Initialize statistics for timing measurements
    // REGISTER_VARIABLE_FOR_WRITING(mpc_solve_time_, "mpc_solve_time");

    // Initialize publishers and subscribers
    sub_current_state_ = nh_.subscribe("/srbd_current", 1, &MPCNode::callbackSrbdCurrent, this);
    pub_mpc_solution_ = nh_.advertise<g1_msgs::SRBD_state>("/mpc_solution", 10);

    ROS_INFO("MPC node initialized successfully");
  }

  void callbackSrbdCurrent(const g1_msgs::SRBD_state::ConstPtr& msg)
  {
    Eigen::VectorXd x0 = mpc_->getX0();
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
    std::vector<std::vector<double>> contact_positions(mpc_->horizon_length_);
    for (int i = 0; i < mpc_->horizon_length_; i++) {
      contact_positions[i].resize(msg->contacts.size() * 3);
      for (size_t j = 0; j < msg->contacts.size(); j++) {
        contact_positions[i][j*3] = msg->contacts[j].position.x;
        contact_positions[i][j*3+1] = msg->contacts[j].position.y;
        contact_positions[i][j*3+2] = msg->contacts[j].position.z;
      }
    }

    // Store contact positions in MPC
    Eigen::MatrixXd c_horizon = mpc_->getCHorizon();
    for (int i = 0; i < mpc_->horizon_length_; i++) {
      for (size_t j = 0; j < msg->contacts.size() * 3; j++) {
        c_horizon(i, j) = contact_positions[i][j];
      }
    }

    // Print c_horizon
    std::cout << "c_horizon: \n" << c_horizon << std::endl;
    mpc_->setCHorizon(c_horizon);

  }

  std::vector<std::vector<int>> gaitPlanner()
  {
    static ros::Time last_switch_time = ros::Time::now();
    static int current_phase = 0;
    const double stance_duration = 3.2; // seconds
    
    // Calculate elapsed time since last switch
    ros::Time current_time = ros::Time::now();
    double elapsed = (current_time - last_switch_time).toSec();
    
    // Check if it's time to switch gait pattern
    if (elapsed >= stance_duration) {
      current_phase = (current_phase + 1) % 2;
      last_switch_time = current_time;
    }
    
    // Create contact horizon based on current phase
    std::vector<std::vector<int>> contact_horizon;
    for (int i = 0; i < mpc_->horizon_length_; i++) {
      if (current_phase == 0) {
        contact_horizon.push_back({1, 1, 0, 0});
      } else {
        contact_horizon.push_back({0, 0, 1, 1});
      }
    }
    
    return contact_horizon;
  }

  void publishMPCSolution()
  {
    g1_msgs::SRBD_state srbd_state_msg;
    srbd_state_msg.header.stamp = ros::Time::now();
    srbd_state_msg.header.frame_id = "SRBD";

    // Get the optimal state trajectory and control inputs
    const Eigen::MatrixXd& x_opt = mpc_->getXOpt();
    const Eigen::MatrixXd& u_opt = mpc_->getUOpt();
    
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
    
    // Get contact information (active/inactive status)
    auto contact_horizon = gaitPlanner();
    
    // Set contact forces from optimized controls
    std::vector<std::string> contact_names = {
      "left_foot_line_contact_lower", 
      "left_foot_line_contact_upper", 
      "right_foot_line_contact_lower", 
      "right_foot_line_contact_upper"
    };
    
    for (int i = 0; i < 4; i++) {
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
    // Get contact horizon (which feet are in contact)
    
    std::vector<std::vector<int>> contact_horizon;
    for (int i = 0; i < mpc_->horizon_length_; i++) {
      contact_horizon.push_back({1, 1, 1, 1}); // All feet in contact
    }
    
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
    for (int i = 0; i < mpc_->horizon_length_; i++) {
      x_ref_hor(i, 3) = x_const;
      x_ref_hor(i, 4) = y_center;
      x_ref_hor(i, 5) = z_center;
      // Uncomment to add oscillation: - radius * std::cos(speed*M_PI*time)
      x_ref_hor(i, 12) = mpc_->g_;
    }
    mpc_->setXRefHor(x_ref_hor);
    
    // Extract COM positions for the horizon
    std::vector<std::vector<double>> p_com_horizon(mpc_->horizon_length_);
    for (int i = 0; i < mpc_->horizon_length_; i++) {
      p_com_horizon[i].resize(mpc_->getX0().size());
      for (int j = 0; j < 3; j++) {
        p_com_horizon[i][j+3] = x_ref_hor(i, j+3);
    }
    }
    
    // Convert c_horizon to vector format
    Eigen::MatrixXd c_horizon_mat = mpc_->getCHorizon();
    std::vector<std::vector<double>> c_horizon(mpc_->horizon_length_);
    for (int i = 0; i < mpc_->horizon_length_; i++) {
      c_horizon[i].resize(c_horizon_mat.cols());
      for (int j = 0; j < c_horizon_mat.cols(); j++) {
        c_horizon[i][j] = c_horizon_mat(i, j);
      }
    }
    
    // If using alternative gait patterns:
    // contact_horizon = gaitPlanner();
    
    mpc_->debugPrintMatrixDimensions();

    // Update MPC solution
    auto start_time = ros::WallTime::now();
    
    
    Eigen::VectorXd x_current = mpc_->getX0();
    mpc_->update(contact_horizon, c_horizon, p_com_horizon, &x_current, true);
    
    mpc_solve_time_ = (ros::WallTime::now() - start_time).toSec() * 1000.0; // ms
    
    // Publish statistics
    
    // PUBLISH_STATISTICS();
    
    // Publish solution
    publishMPCSolution();
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_current_state_;
  ros::Publisher pub_mpc_solution_;
  
  std::shared_ptr<g1_mpc::MPC> mpc_;
  
  double mpc_solve_time_; // For statistics
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "mpc_to_wbid");
  
  MPCNode node;
  
  
  while (ros::ok()) {
    ros::spin();
    node.update();
    }
  
  return 0;
}
