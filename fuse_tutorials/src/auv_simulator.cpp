/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2024, Forssea Robotics
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#include <tf2/LinearMath/Quaternion.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fuse_core/eigen.hpp>
#include <Eigen/Core>
#include <memory>
#include <random>

#include <fuse_core/node_interfaces/node_interfaces.hpp>
#include <fuse_core/util.hpp>
#include <fuse_msgs/srv/set_pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <utility>
#include <utility>

namespace
{
constexpr char baselinkFrame[] = "base_link";         //!< The base_link frame id used when
                                                      //!< publishing sensor data
constexpr char baselinkNavFrame[] = "base_link_nav";  //!< The base_link_nav frame id used when
                                                      //!< publishing depth & altitude sensor data
constexpr char mapFrame[] = "map";                    //!< The map frame id used when publishing ground truth
                                                      //!< data
constexpr char seaSurfaceFrame[] =
    "sea_surface";  //!< The sea surface is defined by a pressure sensor, and will be placed
                    //!< above the base_link frame, with a z offset and a null orientation
constexpr char seaBottomFrame[] =
    "sea_bottom";                               //!< The sea surface is defined by a pressure sensor, and will be placed
                                                //!< above the base_link frame, with a z offset and a null orientation
constexpr double imuSigma = 0.001;              //!< Std dev of simulated Imu measurement noise
constexpr char odomFrame[] = "odom";            //!< The odom frame id used when publishing wheel
constexpr double odomPositionSigma = 0.05;      //!< Std dev of simulated odom position measurement noise
constexpr double odomOrientationSigma = 0.001;  //!< Std dev of simulated odom orientation measurement noise
constexpr double twistLinearSigma = 0.001;      //!< Std dev of simulated twist measurement noise
constexpr double seaSurfaceHeight = 5.;         //!< The sea surface is 10 meters above the ellipsoid surface
constexpr double seaBottomHeight = -5.;         //!< The sea bottom is 10 meters below the ellipsoid surface

constexpr double x_amplitude = 5.0;  //!< The amplitude of the x trajectory
constexpr double y_amplitude = 5.0;  //!< The amplitude of the y trajectory
constexpr double z_amplitude = 0.0;  //!< The amplitude of the z trajectory
constexpr double period = 60.0;      //!< The period of the trajectory
}  // namespace

/**
 * @brief The true pose and velocity of the robot
 * z is the ellipsoid height, depth the distance between the robot and
 * the sea surface (positive underwater), and altitude the distance between
 * the sea bottom and the robot (positive above the sea bottom).
 *
 * In this example, the sea surface is 10 meters above the ellipsoid surface,
 * and the sea bottom is 10 meters below the ellipsoid surface. The robot is
 * initially at the ellipsoid surface.
 */
struct State
{
  State() = default;

  State(rclcpp::Time stamp, Eigen::Vector3d position, Eigen::Quaterniond orientation,
        Eigen::Vector3d linear_velocity_body_frame, Eigen::Vector3d angular_velocity_body_frame,
        Eigen::Vector3d linear_acceleration_body_frame)
    : stamp(std::move(stamp))
    , position(std::move(position))
    , orientation(std::move(orientation))
    , linear_velocity_body_frame(std::move(linear_velocity_body_frame))
    , angular_velocity_body_frame(std::move(angular_velocity_body_frame))
    , linear_acceleration_body_frame(std::move(linear_acceleration_body_frame))
    , depth(seaSurfaceHeight - this->position.z())
    , altitude(-seaBottomHeight + this->position.z())
  {
  }

  rclcpp::Time stamp;

  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d linear_velocity_body_frame = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity_body_frame = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration_body_frame = Eigen::Vector3d::Zero();

  double depth = seaSurfaceHeight;
  double altitude = -seaBottomHeight;
};

/**
 * @brief Convert the robot state into a ground truth odometry message
 */
nav_msgs::msg::Odometry stateToTrueOdometry(State const& state)
{
  nav_msgs::msg::Odometry msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = mapFrame;
  msg.child_frame_id = baselinkFrame;
  msg.pose.pose.position.x = state.position.x();
  msg.pose.pose.position.y = state.position.y();
  msg.pose.pose.position.z = state.position.z();

  msg.pose.pose.orientation.w = state.orientation.w();
  msg.pose.pose.orientation.x = state.orientation.x();
  msg.pose.pose.orientation.y = state.orientation.y();
  msg.pose.pose.orientation.z = state.orientation.z();
  msg.twist.twist.linear.x = state.linear_velocity_body_frame.x();
  msg.twist.twist.linear.y = state.linear_velocity_body_frame.y();
  msg.twist.twist.linear.z = state.linear_velocity_body_frame.z();
  msg.twist.twist.angular.x = state.angular_velocity_body_frame.x();
  msg.twist.twist.angular.y = state.angular_velocity_body_frame.y();
  msg.twist.twist.angular.z = state.angular_velocity_body_frame.z();

  // set covariances
  msg.pose.covariance[0] = 1e-9;
  msg.pose.covariance[7] = 1e-9;
  msg.pose.covariance[14] = 1e-9;
  msg.pose.covariance[21] = 1e-9;
  msg.pose.covariance[28] = 1e-9;
  msg.pose.covariance[35] = 1e-9;
  msg.twist.covariance[0] = 1e-9;
  msg.twist.covariance[7] = 1e-9;
  msg.twist.covariance[14] = 1e-9;
  msg.twist.covariance[21] = 1e-9;
  msg.twist.covariance[28] = 1e-9;
  msg.twist.covariance[35] = 1e-9;
  return msg;
}

sensor_msgs::msg::Imu stateToTrueImu(State const& state)
{
  sensor_msgs::msg::Imu msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = baselinkFrame;

  // measure accel
  msg.linear_acceleration.x = state.linear_acceleration_body_frame.x();
  msg.linear_acceleration.y = state.linear_acceleration_body_frame.y();
  msg.linear_acceleration.z = state.linear_acceleration_body_frame.z();
  msg.linear_acceleration_covariance[0] = 1e-9;
  msg.linear_acceleration_covariance[4] = 1e-9;
  msg.linear_acceleration_covariance[8] = 1e-9;

  msg.orientation.w = state.orientation.w();
  msg.orientation.x = state.orientation.x();
  msg.orientation.y = state.orientation.y();
  msg.orientation.z = state.orientation.z();
  msg.orientation_covariance[0] = 1e-9;
  msg.orientation_covariance[4] = 1e-9;
  msg.orientation_covariance[8] = 1e-9;

  msg.angular_velocity.x = state.angular_velocity_body_frame.x();
  msg.angular_velocity.y = state.angular_velocity_body_frame.y();
  msg.angular_velocity.z = state.angular_velocity_body_frame.z();
  msg.angular_velocity_covariance[0] = 1e-9;
  msg.angular_velocity_covariance[4] = 1e-9;
  msg.angular_velocity_covariance[8] = 1e-9;
  return msg;
}

/**
 * @brief Create a simulated Imu measurement from the current state
 */
sensor_msgs::msg::Imu simulateImu(State const& state)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> noise{ 0.0, imuSigma };

  sensor_msgs::msg::Imu msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = baselinkFrame;

  // measure accel
  msg.linear_acceleration.x = state.linear_acceleration_body_frame.x() + noise(generator);
  msg.linear_acceleration.y = state.linear_acceleration_body_frame.y() + noise(generator);
  msg.linear_acceleration.z = state.linear_acceleration_body_frame.z() + noise(generator);
  msg.linear_acceleration_covariance[0] = imuSigma * imuSigma;
  msg.linear_acceleration_covariance[4] = imuSigma * imuSigma;
  msg.linear_acceleration_covariance[8] = imuSigma * imuSigma;

  msg.orientation.w = state.orientation.w();
  msg.orientation.x = state.orientation.x();
  msg.orientation.y = state.orientation.y();
  msg.orientation.z = state.orientation.z();
  msg.orientation_covariance[0] = imuSigma * imuSigma;
  msg.orientation_covariance[4] = imuSigma * imuSigma;
  msg.orientation_covariance[8] = imuSigma * imuSigma;

  msg.angular_velocity.x = state.angular_velocity_body_frame.x() + noise(generator);
  msg.angular_velocity.y = state.angular_velocity_body_frame.y() + noise(generator);
  msg.angular_velocity.z = state.angular_velocity_body_frame.z() + noise(generator);
  msg.angular_velocity_covariance[0] = imuSigma * imuSigma;
  msg.angular_velocity_covariance[4] = imuSigma * imuSigma;
  msg.angular_velocity_covariance[8] = imuSigma * imuSigma;
  return msg;
}

nav_msgs::msg::Odometry simulateOdometry(const State& state)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> position_noise{ 0.0, odomPositionSigma };

  nav_msgs::msg::Odometry msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = odomFrame;
  msg.child_frame_id = baselinkFrame;

  // noisy position measurement
  msg.pose.pose.position.x = state.position.x() + position_noise(generator);
  msg.pose.pose.position.y = state.position.y() + position_noise(generator);
  msg.pose.pose.position.z = state.position.z() + position_noise(generator);
  msg.pose.covariance[0] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[7] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[14] = odomPositionSigma * odomPositionSigma;

  // noisy orientation measurement
  msg.pose.pose.orientation.w = state.orientation.w();
  msg.pose.pose.orientation.x = state.orientation.x();
  msg.pose.pose.orientation.y = state.orientation.y();
  msg.pose.pose.orientation.z = state.orientation.z();
  msg.pose.covariance[21] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[28] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[35] = odomOrientationSigma * odomOrientationSigma;
  return msg;
}

nav_msgs::msg::Odometry simulateDepth(const State& state)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> position_noise{ 0.0, odomPositionSigma };

  nav_msgs::msg::Odometry msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = baselinkNavFrame;
  msg.child_frame_id = seaSurfaceFrame;

  // noisy position measurement
  msg.pose.pose.position.z = state.depth + position_noise(generator);
  msg.pose.covariance[0] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[7] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[14] = odomPositionSigma * odomPositionSigma;

  // orientation measurement
  msg.pose.pose.orientation.w = 1.0;
  msg.pose.pose.orientation.x = 0.0;
  msg.pose.pose.orientation.y = 0.0;
  msg.pose.pose.orientation.z = 0.0;
  msg.pose.covariance[21] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[28] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[35] = odomOrientationSigma * odomOrientationSigma;
  return msg;
}

nav_msgs::msg::Odometry simulateAltitude(const State& state)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> position_noise{ 0.0, odomPositionSigma };

  nav_msgs::msg::Odometry msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = baselinkNavFrame;
  msg.child_frame_id = seaBottomFrame;

  // noisy position measurement
  msg.pose.pose.position.z = -state.altitude + position_noise(generator);
  msg.pose.covariance[0] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[7] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[14] = odomPositionSigma * odomPositionSigma;

  // orientation measurement
  msg.pose.pose.orientation.w = 1.0;
  msg.pose.pose.orientation.x = 0.0;
  msg.pose.pose.orientation.y = 0.0;
  msg.pose.pose.orientation.z = 0.0;
  msg.pose.covariance[21] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[28] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[35] = odomOrientationSigma * odomOrientationSigma;
  return msg;
}

geometry_msgs::msg::TwistWithCovarianceStamped simulateTwist(const State& state)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> twist_noise{ 0.0, twistLinearSigma };

  geometry_msgs::msg::TwistWithCovarianceStamped msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = baselinkFrame;

  // noisy position measurement
  msg.twist.twist.linear.x = state.linear_velocity_body_frame.x() + twist_noise(generator);
  msg.twist.twist.linear.y = state.linear_velocity_body_frame.y() + twist_noise(generator);
  msg.twist.twist.linear.z = state.linear_velocity_body_frame.z() + twist_noise(generator);
  msg.twist.covariance[0] = twistLinearSigma * twistLinearSigma;
  msg.twist.covariance[7] = twistLinearSigma * twistLinearSigma;
  msg.twist.covariance[14] = twistLinearSigma * twistLinearSigma;
  return msg;
}

void initializeStateEstimation(fuse_core::node_interfaces::NodeInterfaces<ALL_FUSE_CORE_NODE_INTERFACES> interfaces,
                               const State& state, const rclcpp::Clock::SharedPtr& clock, const rclcpp::Logger& logger)
{
  // Send the initial localization signal to the state estimator
  auto srv = std::make_shared<fuse_msgs::srv::SetPose::Request>();
  srv->pose.header.frame_id = mapFrame;
  srv->pose.pose.pose.position.x = state.position.x();
  srv->pose.pose.pose.position.y = state.position.y();
  srv->pose.pose.pose.position.z = state.position.z();
  srv->pose.pose.pose.orientation.w = state.orientation.w();
  srv->pose.pose.pose.orientation.x = state.orientation.x();
  srv->pose.pose.pose.orientation.y = state.orientation.y();
  srv->pose.pose.pose.orientation.z = state.orientation.z();
  srv->pose.pose.covariance[0] = 1.0;
  srv->pose.pose.covariance[7] = 1.0;
  srv->pose.pose.covariance[14] = 1.0;
  srv->pose.pose.covariance[21] = 1.0;
  srv->pose.pose.covariance[28] = 1.0;
  srv->pose.pose.covariance[35] = 1.0;

  auto const client = rclcpp::create_client<fuse_msgs::srv::SetPose>(
      interfaces.get_node_base_interface(), interfaces.get_node_graph_interface(),
      interfaces.get_node_services_interface(), "/state_estimation/set_pose_service",
      rclcpp::ServicesQoS().get_rmw_qos_profile(), interfaces.get_node_base_interface()->get_default_callback_group());

  while (!client->wait_for_service(std::chrono::seconds(30)) &&
         interfaces.get_node_base_interface()->get_context()->is_valid())
  {
    RCLCPP_WARN_STREAM(logger, "Waiting for '" << client->get_service_name() << "' service to become available.");
  }

  auto success = false;
  while (!success)
  {
    clock->sleep_for(std::chrono::milliseconds(100));
    srv->pose.header.stamp = clock->now();
    auto result_future = client->async_send_request(srv);

    if (rclcpp::spin_until_future_complete(interfaces.get_node_base_interface(), result_future,
                                           std::chrono::seconds(1)) != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(logger, "set pose service call failed");
      client->remove_pending_request(result_future);
      return;
    }
    success = result_future.get()->success;
  }
}

State generateTrajectory(double t)
{
  double f = 2 * M_PI / period;
  Eigen::Vector3d path{ x_amplitude * std::cos(t * f), y_amplitude * std::sin(t * f), z_amplitude * std::sin(t * f) };
  Eigen::Vector3d position = path;
  Eigen::Vector3d path_dot{ -x_amplitude * f * std::sin(t * f), y_amplitude * f * std::cos(t * f),
                            z_amplitude * f * std::cos(t * f) };
  Eigen::Vector3d path_dot_dot{ -x_amplitude * f * f * std::cos(t * f), -y_amplitude * f * f * std::sin(t * f),
                                -z_amplitude * f * f * std::sin(t * f) };
  static Eigen::Vector3d unit_x = Eigen::Vector3d::UnitX();
  Eigen::Vector3d unit_vel = path_dot.normalized();
  Eigen::Vector3d unit_vel_xy = unit_vel;
  unit_vel_xy.z() = 0;
  Eigen::Quaterniond orientation = Eigen::Quaterniond::FromTwoVectors(unit_vel_xy, unit_vel) *
                                   Eigen::Quaterniond::FromTwoVectors(unit_x, unit_vel_xy);
  orientation.normalize();
  Eigen::Vector3d velocity = orientation.conjugate() * path_dot;  // in body frame
  Eigen::Vector3d angular_velocity = unit_x.cross(orientation.inverse() * path_dot_dot);
  Eigen::Vector3d acceleration = orientation.conjugate() * path_dot_dot;  // in body frame
  return { rclcpp::Time(), position, orientation, velocity, angular_velocity, acceleration };
}

int main(int argc, char** argv)
{
  // set up our ROS node
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("three_dimensional_simulator");

  // create our sensor publishers
  auto imu_publisher = node->create_publisher<sensor_msgs::msg::Imu>("imu", 1);
  auto odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  auto twist_publisher = node->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("twist", 1);
  auto depth_publisher = node->create_publisher<nav_msgs::msg::Odometry>("depth", 1);
  auto altitude_publisher = node->create_publisher<nav_msgs::msg::Odometry>("altitude", 1);

  // create the ground truth publisher
  auto ground_truth_publisher = node->create_publisher<nav_msgs::msg::Odometry>("ground_truth", 1);
  auto true_imu_publisher = node->create_publisher<sensor_msgs::msg::Imu>("true_imu", 1);

  // you can modify the rate at which this loop runs to see the different performance of the estimator and the effect of
  // integration inaccuracy on the ground truth
  auto rate = rclcpp::Rate(100.0);

  // normally we would have to initialize the state estimation, but we included an ignition 'sensor' in our config,
  // which takes care of that.

  while (rclcpp::ok())
  {
    // store the first time this runs (since it won't start running exactly at a multiple of `motion_duration`)
    static auto const firstTime = node->now();
    auto const now = node->now();

    // compensate for the original time offset
    double const now_d = (now - firstTime).seconds();

    // Simulate the robot motion
    auto state = generateTrajectory(now_d);
    state.stamp = now;

    // Publish the new ground truth
    ground_truth_publisher->publish(stateToTrueOdometry(state));
    true_imu_publisher->publish(stateToTrueImu(state));

    // Generate and publish simulated measurements from the new robot state
    imu_publisher->publish(simulateImu(state));
    odom_publisher->publish(simulateOdometry(state));
    twist_publisher->publish(simulateTwist(state));
    depth_publisher->publish(simulateDepth(state));
    altitude_publisher->publish(simulateAltitude(state));

    // Wait for the next time step
    rclcpp::spin_some(node);
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
