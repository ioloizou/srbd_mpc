from mpc import MPC
import numpy as np

import rospy
from g1_msgs.msg import SRBD_state
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header



if __name__ == '__main__':
	rospy.init_node('mpc_to_wbid', anonymous=True)
		
	MPC = MPC()
	MPC.init()
	

	while not rospy.is_shutdown():	

		# Set always the gravity in reference
		MPC.x_ref_hor[:, -1] = MPC.g 

		MPC.update()
