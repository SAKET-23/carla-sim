cmake_minimum_required(VERSION 3.8)
project(hybrid_av_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FeatureVector.msg"
  "msg/PolicyAction.msg"
  "msg/ReferenceTrajectory.msg"
  "msg/MpcDiagnostics.msg"
  "msg/VehicleCommand.msg"
  "msg/TrafficContext.msg"
  "msg/Track.msg"
  "msg/TrackArray.msg"
  "msg/RouteInfo.msg"
  "srv/LoadPolicy.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_interfaces test/test_interfaces.py)
endif()

ament_package()
