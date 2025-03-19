cmake_minimum_required(VERSION 3.0.2)
project(g1_mpc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  pal_statistics
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/mpc_to_wbid.py
  src/mpc.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  src/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
