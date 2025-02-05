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

namespace
{
constexpr char baselinkFrame[] = "base_link";               //!< The base_link frame id used when
                                                            //!< publishing sensor data
constexpr char mapFrame[] = "map";                          //!< The map frame id used when publishing ground truth
                                                            //!< data
constexpr char seaSurfaceFrame[] = "sea_surface";       //!< The sea surface is defined by a pressure sensor, and will be placed
                                                            //!< above the base_link frame, with a z offset and a null orientation 
constexpr char seaBottomFrame[] = "sea_bottom";             //!< The sea surface is defined by a pressure sensor, and will be placed
                                                            //!< above the base_link frame, with a z offset and a null orientation 
constexpr double imuSigma = 0.001;                          //!< Std dev of simulated Imu measurement noise
constexpr char odomFrame[] = "odom";                        //!< The odom frame id used when publishing wheel
constexpr double odomPositionSigma = 0.05;                  //!< Std dev of simulated odom position measurement noise
constexpr double odomOrientationSigma = 0.001;              //!< Std dev of simulated odom orientation measurement noise
constexpr double twistLinearSigma = 0.001;                  //!< Std dev of simulated twist measurement noise
constexpr double seaSurfaceHeight = 10.;                     //!< The sea surface is 10 meters above the ellipsoid surface
constexpr double seaBottomHeight = -40.;                       //!< The sea bottom is 40 meters below the ellipsoid surface
}  // namespace

/**
 * @brief The true pose and velocity of the robot
 * z is the ellipsoid height, depth the distance between the robot and
 * the sea surface (positive underwater), and altitude the distance between
 * the sea bottom and the robot (positive above the sea bottom).
 *
 * In this example, the sea surface is 10 meters above the ellipsoid surface,
 * and the sea bottom is 40 meters below the ellipsoid surface. The robot is
 * initially at the ellipsoid surface.
 */
struct Robot
{
  rclcpp::Time stamp;

  double mass{};

  double x = 0;
  double y = 0;
  double z = 0;
  double depth = seaSurfaceHeight;
  double altitude = -seaBottomHeight;
  double roll = 0;
  double pitch = 0;
  double yaw = 0;
  double vx = 0;
  double vy = 0;
  double vz = 0;
  double vroll = 0;
  double vpitch = 0;
  double vyaw = 0;
  double ax = 0;
  double ay = 0;
  double az = 0;
};

/**
 * @brief Convert the robot state into a ground truth odometry message
 */
nav_msgs::msg::Odometry robotToOdometry(Robot const& state)
{
  nav_msgs::msg::Odometry msg;
  msg.header.stamp = state.stamp;
  msg.header.frame_id = mapFrame;
  msg.child_frame_id = baselinkFrame;
  msg.pose.pose.position.x = state.x;
  msg.pose.pose.position.y = state.y;
  msg.pose.pose.position.z = state.z;

  tf2::Quaternion q;
  q.setEuler(state.yaw, state.pitch, state.roll);
  msg.pose.pose.orientation.w = q.w();
  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.twist.twist.linear.x = state.vx;
  msg.twist.twist.linear.y = state.vy;
  msg.twist.twist.linear.z = state.vz;
  msg.twist.twist.angular.x = state.vroll;
  msg.twist.twist.angular.y = state.vpitch;
  msg.twist.twist.angular.z = state.vyaw;

  // set covariances
  msg.pose.covariance[0] = 0.1;
  msg.pose.covariance[7] = 0.1;
  msg.pose.covariance[14] = 0.1;
  msg.pose.covariance[21] = 0.1;
  msg.pose.covariance[28] = 0.1;
  msg.pose.covariance[35] = 0.1;
  msg.twist.covariance[0] = 0.1;
  msg.twist.covariance[7] = 0.1;
  msg.twist.covariance[14] = 0.1;
  msg.twist.covariance[21] = 0.1;
  msg.twist.covariance[28] = 0.1;
  msg.twist.covariance[35] = 0.1;
  return msg;
}

/**
 * @brief Compute the next robot state given the current robot state and a simulated step time
 */
Robot simulateRobotMotion(Robot const& previous_state, rclcpp::Time const& now, Eigen::Vector3d const& external_force)
{
  auto const dt = (now - previous_state.stamp).seconds();
  auto next_state = Robot();
  next_state.stamp = now;
  next_state.mass = previous_state.mass;

  // just euler integrate to get the next position and velocity
  next_state.x = previous_state.x + previous_state.vx * dt;
  next_state.y = previous_state.y + previous_state.vy * dt;
  next_state.z = previous_state.z + previous_state.vz * dt;
  next_state.depth = seaSurfaceHeight - next_state.z;
  next_state.altitude = -seaBottomHeight + next_state.z;
  next_state.vx = previous_state.vx + previous_state.ax * dt;
  next_state.vy = previous_state.vy + previous_state.ay * dt;
  next_state.vz = previous_state.vz + previous_state.az * dt;

  // let's not deal with 3D orientation dynamics for this tutorial
  next_state.roll = 0;
  next_state.pitch = 0;
  next_state.yaw = 0;
  next_state.vroll = 0;
  next_state.vpitch = 0;
  next_state.vyaw = 0;

  // get the current acceleration from the current applied force
  next_state.ax = external_force.x() / next_state.mass;
  next_state.ay = external_force.y() / next_state.mass;
  next_state.az = external_force.z() / next_state.mass;

  return next_state;
}

/**
 * @brief Create a simulated Imu measurement from the current state
 */
sensor_msgs::msg::Imu simulateImu(Robot const& robot)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> noise{ 0.0, imuSigma };

  sensor_msgs::msg::Imu msg;
  msg.header.stamp = robot.stamp;
  msg.header.frame_id = baselinkFrame;

  // measure accel
  msg.linear_acceleration.x = robot.ax + noise(generator);
  msg.linear_acceleration.y = robot.ay + noise(generator);
  msg.linear_acceleration.z = robot.az + noise(generator);
  msg.linear_acceleration_covariance[0] = imuSigma * imuSigma;
  msg.linear_acceleration_covariance[4] = imuSigma * imuSigma;
  msg.linear_acceleration_covariance[8] = imuSigma * imuSigma;

  // Simulated IMU does not provide orientation (negative covariance indicates this)
  msg.orientation_covariance[0] = -1;
  msg.orientation_covariance[4] = -1;
  msg.orientation_covariance[8] = -1;

  msg.angular_velocity.x = robot.vroll + noise(generator);
  msg.angular_velocity.y = robot.vpitch + noise(generator);
  msg.angular_velocity.z = robot.vyaw + noise(generator);
  msg.angular_velocity_covariance[0] = imuSigma * imuSigma;
  msg.angular_velocity_covariance[4] = imuSigma * imuSigma;
  msg.angular_velocity_covariance[8] = imuSigma * imuSigma;
  return msg;
}

nav_msgs::msg::Odometry simulateOdometry(const Robot& robot)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> position_noise{ 0.0, odomPositionSigma };

  nav_msgs::msg::Odometry msg;
  msg.header.stamp = robot.stamp;
  msg.header.frame_id = odomFrame;
  msg.child_frame_id = baselinkFrame;

  // noisy position measurement
  msg.pose.pose.position.x = robot.x + position_noise(generator);
  msg.pose.pose.position.y = robot.y + position_noise(generator);
  msg.pose.pose.position.z = robot.z + position_noise(generator);
  msg.pose.covariance[0] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[7] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[14] = odomPositionSigma * odomPositionSigma;

  // noisy orientation measurement
  tf2::Quaternion q;
  q.setEuler(robot.yaw, robot.pitch, robot.roll);
  msg.pose.pose.orientation.w = q.w();
  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.pose.covariance[21] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[28] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[35] = odomOrientationSigma * odomOrientationSigma;
  return msg;
}

nav_msgs::msg::Odometry simulateDepth(const Robot& robot)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> position_noise{ 0.0, odomPositionSigma };

  nav_msgs::msg::Odometry msg;
  msg.header.stamp = robot.stamp;
  msg.header.frame_id = baselinkFrame;
  msg.child_frame_id = seaSurfaceFrame;

  // noisy position measurement
  msg.pose.pose.position.z = robot.depth + position_noise(generator);
  msg.pose.covariance[0] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[7] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[14] = odomPositionSigma * odomPositionSigma;

  // noisy orientation measurement
  tf2::Quaternion q;
  q.setEuler(robot.yaw, robot.pitch, robot.roll);
  msg.pose.pose.orientation.w = -q.w();
  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.pose.covariance[21] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[28] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[35] = odomOrientationSigma * odomOrientationSigma;
  return msg;
}

nav_msgs::msg::Odometry simulateAltitude(const Robot& robot)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> position_noise{ 0.0, odomPositionSigma };

  nav_msgs::msg::Odometry msg;
  msg.header.stamp = robot.stamp;
  msg.header.frame_id = baselinkFrame;
  msg.child_frame_id = seaBottomFrame;

  // noisy position measurement
  msg.pose.pose.position.z = -robot.altitude + position_noise(generator);
  msg.pose.covariance[0] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[7] = odomPositionSigma * odomPositionSigma;
  msg.pose.covariance[14] = odomPositionSigma * odomPositionSigma;

  // noisy orientation measurement
  tf2::Quaternion q;
  q.setEuler(robot.yaw, robot.pitch, robot.roll);
  msg.pose.pose.orientation.w = -q.w();
  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.pose.covariance[21] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[28] = odomOrientationSigma * odomOrientationSigma;
  msg.pose.covariance[35] = odomOrientationSigma * odomOrientationSigma;
  return msg;
}

geometry_msgs::msg::TwistWithCovarianceStamped simulateTwist(const Robot& robot)
{
  static std::random_device rd{};
  static std::mt19937 generator{ rd() };
  static std::normal_distribution<> twist_noise{ 0.0, twistLinearSigma };

  geometry_msgs::msg::TwistWithCovarianceStamped msg;
  msg.header.stamp = robot.stamp;
  msg.header.frame_id = mapFrame;

  // noisy position measurement
  msg.twist.twist.linear.x = robot.vx + twist_noise(generator);
  msg.twist.twist.linear.y = robot.vy + twist_noise(generator);
  msg.twist.twist.linear.z = robot.vz + twist_noise(generator);
  msg.twist.covariance[0] = twistLinearSigma * twistLinearSigma;
  msg.twist.covariance[7] = twistLinearSigma * twistLinearSigma;
  msg.twist.covariance[14] = twistLinearSigma * twistLinearSigma;
  return msg;
}

void initializeStateEstimation(fuse_core::node_interfaces::NodeInterfaces<ALL_FUSE_CORE_NODE_INTERFACES> interfaces,
                               const Robot& state, const rclcpp::Clock::SharedPtr& clock, const rclcpp::Logger& logger)
{
  // Send the initial localization signal to the state estimator
  auto srv = std::make_shared<fuse_msgs::srv::SetPose::Request>();
  srv->pose.header.frame_id = mapFrame;
  srv->pose.pose.pose.position.x = state.x;
  srv->pose.pose.pose.position.y = state.y;
  srv->pose.pose.pose.position.z = state.z;
  tf2::Quaternion q;
  q.setEuler(state.yaw, state.pitch, state.roll);
  srv->pose.pose.pose.orientation.w = q.w();
  srv->pose.pose.pose.orientation.x = q.x();
  srv->pose.pose.pose.orientation.y = q.y();
  srv->pose.pose.pose.orientation.z = q.z();
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

  // Initialize the robot state (state variables are zero-initialized)
  auto state = Robot();
  state.stamp = node->now();
  state.mass = 10;  // kg

  // you can modify the rate at which this loop runs to see the different performance of the estimator and the effect of
  // integration inaccuracy on the ground truth
  auto rate = rclcpp::Rate(10.0);

  // normally we would have to initialize the state estimation, but we included an ignition 'sensor' in our config,
  // which takes care of that.

  // parameters that control the motion pattern of the robot
  double const motion_duration = 5;  // length of time to oscillate on a given axis, in seconds
  double const n_cycles = 2;         // number of oscillations per `motion_duration`

  while (rclcpp::ok())
  {
    // store the first time this runs (since it won't start running exactly at a multiple of `motion_duration`)
    static auto const firstTime = node->now();
    auto const now = node->now();

    // compensate for the original time offset
    double const now_d = (now - firstTime).seconds();
    // store how long it has been (resetting at `motion_duration` seconds)
    double const mod_time = std::fmod(now_d, motion_duration);

    // apply a harmonic force (oscillates `N_cycles` times per `motion_duration`)
    double const force_magnitude = 100 * std::cos(2 * M_PI * n_cycles * mod_time / motion_duration);
    Eigen::Vector3d external_force = { 0, 0, 0 };

    // switch oscillation axes every `motion_duration` seconds (with one 'rest period')
    if (std::fmod(now_d, 4 * motion_duration) < motion_duration)
    {
      external_force.x() = force_magnitude;
    }
    else if (std::fmod(now_d, 4 * motion_duration) < 2 * motion_duration)
    {
      external_force.y() = force_magnitude;
    }
    else if (std::fmod(now_d, 4 * motion_duration) < 3 * motion_duration)
    {
      external_force.z() = force_magnitude;
    }
    else
    {
      // reset the robot's position and velocity, leave the external force as 0
      // we do this so the ground truth doesn't drift (due to inaccuracy from euler integration)
      state.x = 0;
      state.y = 0;
      state.z = 0;
      state.vx = 0;
      state.vy = 0;
      state.vz = 0;
    }

    // Simulate the robot motion
    auto new_state = simulateRobotMotion(state, now, external_force);

    // Publish the new ground truth
    ground_truth_publisher->publish(robotToOdometry(new_state));

    // Generate and publish simulated measurements from the new robot state
    imu_publisher->publish(simulateImu(new_state));
    odom_publisher->publish(simulateOdometry(new_state));
    twist_publisher->publish(simulateTwist(new_state));
    depth_publisher->publish(simulateDepth(new_state));
    altitude_publisher->publish(simulateAltitude(new_state));

    // Wait for the next time step
    state = new_state;
    rclcpp::spin_some(node);
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
