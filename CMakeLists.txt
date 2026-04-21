cmake_minimum_required(VERSION 3.20)
project(pose_evaluator)

set(msg_files
  "msg/DetectedPoint.msg"
  "msg/DetectedPoints.msg"
  "msg/CameraImageStamped.msg"
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV 4.6.0 EXACT REQUIRED )
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(camera_info_manager REQUIRED)

include_directories(include)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs sensor_msgs
)
ament_export_dependencies(rosidl_default_runtime)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

add_executable(image_publisher src/image_publisher/image_publisher.cpp)
target_link_libraries(
  image_publisher PUBLIC 
  rclcpp::rclcpp 
  ${std_msgs_TARGETS} 
  ${sensor_msgs_TARGETS} 
  ${OpenCV_LIBS} 
  cv_bridge::cv_bridge 
  camera_info_manager::camera_info_manager 
  "${cpp_typesupport_target}")

add_executable(image_stream_publisher src/image_publisher/image_stream_publisher.cpp)
target_link_libraries(
  image_stream_publisher PUBLIC 
  rclcpp::rclcpp 
  ${std_msgs_TARGETS} 
  ${sensor_msgs_TARGETS} 
  ${OpenCV_LIBS} 
  cv_bridge::cv_bridge 
  camera_info_manager::camera_info_manager 
  "${cpp_typesupport_target}")
 
add_executable(image_extractor src/image_extractor/image_extractor.cpp)
target_link_libraries(
  image_extractor PUBLIC 
  rclcpp::rclcpp 
  ${std_msgs_TARGETS} 
  ${sensor_msgs_TARGETS} 
  ${OpenCV_LIBS} 
  cv_bridge::cv_bridge 
  camera_info_manager::camera_info_manager 
  "${cpp_typesupport_target}")

add_executable(latest_image_publisher_node src/latest_image_publisher/latest_image_publisher_node.cpp)
target_link_libraries(
  latest_image_publisher_node PUBLIC 
  rclcpp::rclcpp 
  ${std_msgs_TARGETS} 
  ${sensor_msgs_TARGETS} 
  ${OpenCV_LIBS} 
  cv_bridge::cv_bridge 
  camera_info_manager::camera_info_manager 
  "${cpp_typesupport_target}")

add_executable(aruco_detector_node src/aruco_detector/aruco_detector_node.cpp)
target_link_libraries(aruco_detector_node PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS} ${sensor_msgs_TARGETS} ${OpenCV_LIBS} cv_bridge::cv_bridge "${cpp_typesupport_target}")

add_executable(pose_evaluator_node src/pose_evaluator/pose_evaluator_node.cpp)
target_link_libraries(pose_evaluator_node PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS} ${sensor_msgs_TARGETS} ${OpenCV_LIBS} cv_bridge::cv_bridge "${cpp_typesupport_target}" ${tf2_TARGETS} ${tf2_ros_TARGETS} Eigen3::Eigen)

install(TARGETS 
  image_publisher
  image_stream_publisher
  image_extractor
  latest_image_publisher_node
  aruco_detector_node
  pose_evaluator_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
