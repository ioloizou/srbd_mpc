#!/usr/bin/env python3
from mpc import MPC
import numpy as np

import rospy
from g1_msgs.msg import SRBD_state, ContactPoint
from std_msgs.msg import Header

def callback_srbd_current(msg):
	"""
	Callback function to update the current state of the robot.
	"""
	MPC.x0[0] = msg.orientation.x
	MPC.x0[1] = msg.orientation.y
	MPC.x0[2] = msg.orientation.z
	MPC.x0[3] = msg.position.x
	MPC.x0[4] = msg.position.y
	MPC.x0[5] = msg.position.z
	MPC.x0[6] = msg.angular_velocity.x
	MPC.x0[7] = msg.angular_velocity.y
	MPC.x0[8] = msg.angular_velocity.z
	MPC.x0[9] = msg.linear_velocity.x
	MPC.x0[10] = msg.linear_velocity.y
	MPC.x0[11] = msg.linear_velocity.z
	MPC.x0[12] = msg.gravity

	# Get current position of the contact points
	# Until footstep is implemented, we assume that the contact points are constant
	contact_positions = []
	for i in range(len(msg.contacts)):
		contact_positions.extend([msg.contacts[i].position.x,
								  msg.contacts[i].position.y,
								  msg.contacts[i].position.z])
	MPC.c_horizon[:] = np.tile(np.array(contact_positions), (MPC.HORIZON_LENGTH, 1))

def publish_mpc_solution():	
	"""Publish the MPC solution u_opt0 and x_opt1 to the ROS topic."""
	srbd_state_msg = SRBD_state()
	srbd_state_msg.contacts = []
	srbd_state_msg.header = Header()
	srbd_state_msg.header.stamp = rospy.Time.now()
	srbd_state_msg.header.frame_id = "SRBD"
	srbd_state_msg.orientation.x = MPC.x_opt[1, 0]
	srbd_state_msg.orientation.y = MPC.x_opt[1, 1]
	srbd_state_msg.orientation.z = MPC.x_opt[1, 2]
	srbd_state_msg.position.x = MPC.x_opt[1, 3]
	srbd_state_msg.position.y = MPC.x_opt[1, 4]
	srbd_state_msg.position.z = MPC.x_opt[1, 5]
	srbd_state_msg.angular_velocity.x = MPC.x_opt[1, 6]
	srbd_state_msg.angular_velocity.y = MPC.x_opt[1, 7]
	srbd_state_msg.angular_velocity.z = MPC.x_opt[1, 8]
	srbd_state_msg.linear_velocity.x = MPC.x_opt[1, 9]
	srbd_state_msg.linear_velocity.y = MPC.x_opt[1, 10]
	srbd_state_msg.linear_velocity.z = MPC.x_opt[1, 11]
	srbd_state_msg.gravity = MPC.x_opt[1, 12]

	for i, contact_name in enumerate(["left_foot_line_contact_lower", "left_foot_line_contact_upper", "right_foot_line_contact_lower", "right_foot_line_contact_upper"]):
		contact_point_msg = ContactPoint()
		contact_point_msg.name = contact_name
		contact_point_msg.force.x = MPC.u_opt[0, i * 3]
		contact_point_msg.force.y = MPC.u_opt[0, i * 3 + 1]
		contact_point_msg.force.z = MPC.u_opt[0, i * 3 + 2]
		
		# Contact State
		contact_point_msg.active = True
		srbd_state_msg.contacts.append(contact_point_msg)
	pub_mpc_solution.publish(srbd_state_msg)		

if __name__ == '__main__':
	rospy.init_node('mpc', anonymous=True)
		
	MPC = MPC()
	MPC.init_matrices()
	
	sub_current_state = rospy.Subscriber('/srbd_current', SRBD_state, callback_srbd_current)
	pub_mpc_solution = rospy.Publisher('/mpc_solution', SRBD_state, queue_size=10)
	
	while not rospy.is_shutdown():	

		# Temporary until refactor of the code
		c_horizon = MPC.c_horizon.copy()
		
		contact_horizon = []
		# Both feet in contact for all the horizon
		for i in range(MPC.HORIZON_LENGTH):
			contact_horizon.append(np.array([1, 1, 1, 1]))

		p_com_horizon = np.tile([5.26790425e-02, 7.44339342e-05, 5.97983255e-01] , (MPC.HORIZON_LENGTH, 1))

		MPC.x_ref_hor[:, 3:6] = np.tile([5.26790425e-02, 7.44339342e-05, 5.97983255e-01], (MPC.HORIZON_LENGTH, 1))
		MPC.x_ref_hor[:, 12]= MPC.g
		# exit()
		# Update the MPC solution		
		MPC.update(contact_horizon, c_horizon, p_com_horizon, x_current = MPC.x0.copy() , one_rollout = True)
		# Publish the MPC solution
		publish_mpc_solution()
