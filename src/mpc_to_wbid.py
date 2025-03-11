#!/usr/bin/env python3
from mpc import MPC
import numpy as np

import rospy
from g1_msgs.msg import SRBD_state
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header

def callback_srbd_current(msg):
	"""
	Callback function to update the current state of the robot.
	"""
	MPC.x_curr[0] = msg.orientation.x
	MPC.x_curr[1] = msg.orientation.y
	MPC.x_curr[2] = msg.orientation.z
	MPC.x_curr[3] = msg.position.x
	MPC.x_curr[4] = msg.position.y
	MPC.x_curr[5] = msg.position.z
	MPC.x_curr[6] = msg.angular_velocity.x
	MPC.x_curr[7] = msg.angular_velocity.y
	MPC.x_curr[8] = msg.angular_velocity.z
	MPC.x_curr[9] = msg.linear_velocity.x
	MPC.x_curr[10] = msg.linear_velocity.y
	MPC.x_curr[11] = msg.linear_velocity.z
	MPC.x_curr[12] = msg.gravity

	# Get current position of the contact points
	# Until footstep is implemented, we assume that the contact points are constat
	MPC.c_horizon[:] = np.array([msg.contacts[i].position for i in range(len(msg.contacts))])

def publish_mpc_solution():	
	"""Publish the MPC solution u_opt0 and x_opt1 to the ROS topic."""
	srbd_state_msg = SRBD_state()
	srbd_state_msg.contacts = []
	srbd_state_msg.header = Header()
	srbd_state_msg.header.stamp = rospy.Time.now()
	srbd_state_msg.header.frame_id = "SRBD"
	srbd_state_msg.orientation.x = MPC.x_opt1[0]
	srbd_state_msg.orientation.y = MPC.x_opt1[1]
	srbd_state_msg.orientation.z = MPC.x_opt1[2]
	srbd_state_msg.position.x = MPC.x_opt1[3]
	srbd_state_msg.position.y = MPC.x_opt1[4]
	srbd_state_msg.position.z = MPC.x_opt1[5]
	srbd_state_msg.angular_velocity.x = MPC.x_opt1[6]
	srbd_state_msg.angular_velocity.y = MPC.x_opt1[7]
	srbd_state_msg.angular_velocity.z = MPC.x_opt1[8]
	srbd_state_msg.linear_velocity.x = MPC.x_opt1[9]
	srbd_state_msg.linear_velocity.y = MPC.x_opt1[10]
	srbd_state_msg.linear_velocity.z = MPC.x_opt1[11]
	srbd_state_msg.gravity = MPC.x_opt1[12]

	for i in range(len(MPC.u_opt0) // 3):
		# Ground Reaction Force
		contact_point_msg = Vector3()
		contact_point_msg.x = MPC.u_opt0[i * 3]
		contact_point_msg.y = MPC.u_opt0[i * 3 + 1]
		contact_point_msg.z = MPC.u_opt0[i * 3 + 2]
		
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

		
		# Update the MPC solution		
		MPC.update()
		# Publish the MPC solution
		publish_mpc_solution()
