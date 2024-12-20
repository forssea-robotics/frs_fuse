/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2024, PickNik Robotics
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
#ifndef FUSE_MODELS__TRANSFORM_SENSOR_HPP_
#define FUSE_MODELS__TRANSFORM_SENSOR_HPP_

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>

#include <fuse_models/parameters/transform_sensor_params.hpp>
#include <fuse_core/throttled_callback.hpp>

#include <fuse_core/async_sensor_model.hpp>
#include <fuse_core/uuid.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace fuse_models
{

/**
 * @brief An adapter-type sensor that produces pose constraints from published transforms
 *
 * This sensor subscribes to a MessageType topic and creates orientation and pose variables and constraints.
 * This sensor can be used for AprilTags or any pose for which the transform to the desired state estimation frame is
 * known. For an example, try `ros2 launch fuse_tutorials fuse_apriltag_tutorial.launch.py` and see its relevant files.
 *
 * Parameters:
 *  - device_id (uuid string, default: 00000000-0000-0000-0000-000000000000) The device/robot ID to
 *                                                                           publish
 *  - device_name (string) Used to generate the device/robot ID if the device_id is not provided
 *  - queue_size (int, default: 10) The subscriber queue size for the transform messages
 *  - topic (string) The topic to which to subscribe for the transform messages
 *  - target_frame (string) the state estimation frame to transform tfs to
 *
 * Subscribes:
 *  - \p topic (MessageType) IMU data at a given timestep
 */
class TransformSensor : public fuse_core::AsyncSensorModel
{
public:
  FUSE_SMART_PTR_DEFINITIONS(TransformSensor)
  using ParameterType = parameters::TransformSensorParams;
  using MessageType = tf2_msgs::msg::TFMessage;

  /**
   * @brief Default constructor
   */
  TransformSensor();

  /**
   * @brief Destructor
   */
  virtual ~TransformSensor() = default;

  TransformSensor(TransformSensor const&) = delete;
  TransformSensor(TransformSensor&&) = delete;
  TransformSensor& operator=(TransformSensor const&) = delete;
  TransformSensor& operator=(TransformSensor&&) = delete;

  /**
   * @brief Shadowing extension to the AsyncSensorModel::initialize call
   */
  void initialize(fuse_core::node_interfaces::NodeInterfaces<ALL_FUSE_CORE_NODE_INTERFACES> interfaces,
                  const std::string& name, fuse_core::TransactionCallback transaction_callback) override;

  /**
   * @brief Callback for tf messages
   * @param[in] msg - The IMU message to process
   */
  void process(const MessageType& msg);

protected:
  fuse_core::UUID device_id_;  //!< The UUID of this device

  /**
   * @brief Perform any required initialization for the sensor model
   *
   * This could include things like reading from the parameter server or subscribing to topics. The
   * class's node handles will be properly initialized before SensorModel::onInit() is called.
   * Spinning of the callback queue will not begin until after the call to SensorModel::onInit()
   * completes.
   */
  void onInit() override;

  /**
   * @brief Subscribe to the input topic to start sending transactions to the optimizer
   */
  void onStart() override;

  /**
   * @brief Unsubscribe from the input topic to stop sending transactions to the optimizer
   */
  void onStop() override;

  fuse_core::node_interfaces::NodeInterfaces<fuse_core::node_interfaces::Base, fuse_core::node_interfaces::Clock,
                                             fuse_core::node_interfaces::Logging,
                                             fuse_core::node_interfaces::Parameters, fuse_core::node_interfaces::Topics,
                                             fuse_core::node_interfaces::Waitables>
      interfaces_;  //!< Shadows AsyncSensorModel interfaces_

  rclcpp::Clock::SharedPtr clock_;  //!< The sensor model's clock, for timestamping and logging
  rclcpp::Logger logger_;           //!< The sensor model's logger

  ParameterType params_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::set<std::string> transforms_of_interest_;

  rclcpp::Subscription<MessageType>::SharedPtr sub_;

  using AprilTagThrottledCallback = fuse_core::ThrottledMessageCallback<MessageType>;
  AprilTagThrottledCallback throttled_callback_;
};

}  // namespace fuse_models

#endif  // FUSE_MODELS__TRANSFORM_SENSOR_HPP_
