cmake_minimum_required(VERSION 3.0.2)
project(g1_mpc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    geometry_msgs
    pal_statistics
    roscpp
    g1_msgs
)

find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(pybind11 REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES g1_mpc
    CATKIN_DEPENDS roscpp std_msgs geometry_msgs g1_msgs
)

# Include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Build the MPC library
add_library(g1_mpc
    src/mpc.cpp
)

# Link against OSQP
target_link_libraries(g1_mpc
    ${catkin_LIBRARIES}
    osqp::osqp
)

# Python bindings with pybind11
pybind11_add_module(g1_mpc_py src/mpc_python.cpp)
target_link_libraries(g1_mpc_py PRIVATE g1_mpc osqp::osqp)

# C++ node for mpc_to_wbid
add_executable(mpc_to_wbid_node src/mpc_to_wbid_node.cpp)
target_link_libraries(mpc_to_wbid_node
    g1_mpc
    ${catkin_LIBRARIES}
)

# Install targets
install(TARGETS g1_mpc mpc_to_wbid_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install Python module
install(TARGETS g1_mpc_py
    DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}
)

install(FILES
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/__init__.py
    DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}
)

install(DIRECTORY include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp"
)

catkin_install_python(PROGRAMS
    scripts/mpc_to_wbid.py
    scripts/mpc.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
    scripts/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
