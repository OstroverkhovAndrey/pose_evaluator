#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pose_evaluator/msg/detected_points.hpp"
#include "pose_evaluator/filter_factory.hpp"
#include "pose_evaluator/random_accel_process_model.hpp"
#include "pose_evaluator/camera_pinhole_measurement_model.hpp"
#include "pose_evaluator/object_pinhole_measurement_model.hpp"
#include "pose_evaluator/so3_utils.hpp"

using pose_evaluator::State;
using pose_evaluator::Cov12;
using pose_evaluator::IFilter;
using pose_evaluator::CameraIntrinsics;
using pose_evaluator::WorldPointObservation;
using pose_evaluator::ObjectPointObservation;
using pose_evaluator::CameraPinholeMeasurementModel;
using pose_evaluator::ObjectPinholeMeasurementModel;
using pose_evaluator::makeCameraFilter;
using pose_evaluator::makeObjectFilter;
using pose_evaluator::RandomAccelProcessModel;

class PoseEvaluatorNode : public rclcpp::Node
{
public:
  PoseEvaluatorNode()
  : Node("pose_evaluator_node")
  {
    camera_filter_type_ = declare_parameter<std::string>("camera_filter_type", "ukf");
    object_filter_type_ = declare_parameter<std::string>("object_filter_type", "ukf");

    sigma_a_ = declare_parameter<double>("sigma_a", 0.5);
    sigma_alpha_ = declare_parameter<double>("sigma_alpha", 0.5);
    sigma_px_ = declare_parameter<double>("sigma_px", 1.5);

    detections_sub_ = create_subscription<pose_evaluator::msg::DetectedPoints>(
      "detected_points", 10,
      std::bind(&PoseEvaluatorNode::detectionsCallback, this, std::placeholders::_1));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("estimated_pose", 10);
    twist_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("estimated_twist", 10);
  }

private:
  struct TrackedEntity
  {
    std::unique_ptr<IFilter> filter;
    rclcpp::Time last_stamp{0, 0, RCL_ROS_TIME};

    Eigen::Vector3d last_position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond last_orientation{1.0, 0.0, 0.0, 0.0};
    rclcpp::Time last_output_stamp{0, 0, RCL_ROS_TIME};
    bool last_output_valid{false};
  };

  TrackedEntity & getOrCreateCameraEntity(const std::string & camera_id)
  {
    auto it = camera_entities_.find(camera_id);
    if (it != camera_entities_.end()) {
      return it->second;
    }

    auto process_model = std::make_shared<RandomAccelProcessModel>();
    TrackedEntity entity;
    entity.filter = makeCameraFilter(camera_filter_type_, process_model);
    auto [new_it, inserted] = camera_entities_.emplace(camera_id, std::move(entity));
    (void)inserted;
    return new_it->second;
  }

  TrackedEntity & getOrCreateObjectEntity(const std::string & object_id)
  {
    auto it = object_entities_.find(object_id);
    if (it != object_entities_.end()) {
      return it->second;
    }

    auto process_model = std::make_shared<RandomAccelProcessModel>();
    TrackedEntity entity;
    entity.filter = makeObjectFilter(object_filter_type_, process_model);
    auto [new_it, inserted] = object_entities_.emplace(object_id, std::move(entity));
    (void)inserted;
    return new_it->second;
  }

  static CameraIntrinsics intrinsicsFromCameraInfo(const sensor_msgs::msg::CameraInfo & info)
  {
    CameraIntrinsics K;
    K.fx = info.k[0];
    K.fy = info.k[4];
    K.cx = info.k[2];
    K.cy = info.k[5];
    return K;
  }


  bool initializeCameraFilterFromPnP(
  TrackedEntity & entity,
  const CameraIntrinsics & K,
  const std::vector<WorldPointObservation> & observations)
{
  if (observations.size() < 4) {
    return false;
  }

  std::vector<cv::Point3f> world_points;
  std::vector<cv::Point2f> image_points;
  world_points.reserve(observations.size());
  image_points.reserve(observations.size());

  for (const auto & obs : observations) {
    world_points.emplace_back(
      static_cast<float>(obs.point_world.x()),
      static_cast<float>(obs.point_world.y()),
      static_cast<float>(obs.point_world.z()));
    image_points.emplace_back(
      static_cast<float>(obs.pixel.x()),
      static_cast<float>(obs.pixel.y()));
  }

  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
    K.fx, 0.0, K.cx,
    0.0, K.fy, K.cy,
    0.0, 0.0, 1.0);

  cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
  cv::Mat rvec, tvec;

  const bool ok = cv::solvePnP(world_points, image_points, camera_matrix, dist, rvec, tvec);
  if (!ok) {
    return false;
  }

  cv::Mat Rcw_cv;
  cv::Rodrigues(rvec, Rcw_cv);

  Eigen::Matrix3d R_cw;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R_cw(r, c) = Rcw_cv.at<double>(r, c);
    }
  }

  Eigen::Vector3d t_cw(
    tvec.at<double>(0, 0),
    tvec.at<double>(1, 0),
    tvec.at<double>(2, 0));

  const Eigen::Matrix3d R_wc = R_cw.transpose();
  const Eigen::Vector3d p_wc = -R_wc * t_cw;

  State x0;
  x0.p = p_wc;
  x0.q = Eigen::Quaterniond(R_wc).normalized();
  x0.v.setZero();
  x0.w.setZero();

  Cov12 P0 = Cov12::Identity();
  P0.block<3,3>(0,0) *= 1e-2;
  P0.block<3,3>(3,3) *= 1.0;
  P0.block<3,3>(6,6) *= 1e-2;
  P0.block<3,3>(9,9) *= 1.0;

  entity.filter->initialize(x0, P0);
  return true;
}

  
bool initializeObjectFilterFromPnP(
  TrackedEntity & entity,
  const CameraIntrinsics & K,
  const State & camera_state,
  const std::vector<ObjectPointObservation> & observations)
{
  if (observations.size() < 4) {
    return false;
  }

  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(observations.size());
  image_points.reserve(observations.size());

  for (const auto & obs : observations) {
    object_points.emplace_back(
      static_cast<float>(obs.point_object.x()),
      static_cast<float>(obs.point_object.y()),
      static_cast<float>(obs.point_object.z()));
    image_points.emplace_back(
      static_cast<float>(obs.pixel.x()),
      static_cast<float>(obs.pixel.y()));
  }

  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
    K.fx, 0.0, K.cx,
    0.0, K.fy, K.cy,
    0.0, 0.0, 1.0);

  cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
  cv::Mat rvec, tvec;

  const bool ok = cv::solvePnP(object_points, image_points, camera_matrix, dist, rvec, tvec);
  if (!ok) {
    return false;
  }

  cv::Mat Rco_cv;
  cv::Rodrigues(rvec, Rco_cv);

  Eigen::Matrix3d R_co;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R_co(r, c) = Rco_cv.at<double>(r, c);
    }
  }

  Eigen::Vector3d t_co(
    tvec.at<double>(0, 0),
    tvec.at<double>(1, 0),
    tvec.at<double>(2, 0));

  const Eigen::Matrix3d R_wc = camera_state.q.toRotationMatrix();
  const Eigen::Matrix3d R_wo = R_wc * R_co;
  const Eigen::Vector3d p_wo = camera_state.p + R_wc * t_co;

  State x0;
  x0.p = p_wo;
  x0.q = Eigen::Quaterniond(R_wo).normalized();
  x0.v.setZero();
  x0.w.setZero();

  Cov12 P0 = Cov12::Identity();
  P0.block<3,3>(0,0) *= 1e-2;
  P0.block<3,3>(3,3) *= 1.0;
  P0.block<3,3>(6,6) *= 1e-2;
  P0.block<3,3>(9,9) *= 1.0;

  entity.filter->initialize(x0, P0);
  return true;
}
  

void updateCameraEntity(
  TrackedEntity & entity,
  const CameraIntrinsics & K,
  const std::vector<WorldPointObservation> & observations,
  const rclcpp::Time & stamp)
{
  if (observations.size() < 4) {
    return;
  }

  if (!entity.filter->isInitialized()) {
    if (initializeCameraFilterFromPnP(entity, K, observations)) {
      entity.last_stamp = stamp;
    }
    return;
  }

  double dt = 0.0;
  if (entity.last_stamp.nanoseconds() != 0) {
    dt = (stamp - entity.last_stamp).seconds();
    if (dt < 0.0) {
      dt = 0.0;
    }
  }
  entity.last_stamp = stamp;

  entity.filter->predict(dt);

  CameraPinholeMeasurementModel model(K, observations, sigma_px_);
  const Eigen::VectorXd z = model.measurementVector();
  entity.filter->update(z, model);
}


void updateObjectEntity(
  TrackedEntity & entity,
  const CameraIntrinsics & K,
  const State & camera_state,
  const std::vector<ObjectPointObservation> & observations,
  const rclcpp::Time & stamp)
{
  if (observations.size() < 4) {
    return;
  }

  if (!entity.filter->isInitialized()) {
    if (initializeObjectFilterFromPnP(entity, K, camera_state, observations)) {
      entity.last_stamp = stamp;
    }
    return;
  }

  double dt = 0.0;
  if (entity.last_stamp.nanoseconds() != 0) {
    dt = (stamp - entity.last_stamp).seconds();
    if (dt < 0.0) {
      dt = 0.0;
    }
  }
  entity.last_stamp = stamp;

  entity.filter->predict(dt);

  ObjectPinholeMeasurementModel model(K, camera_state, observations, sigma_px_);
  const Eigen::VectorXd z = model.measurementVector();
  entity.filter->update(z, model);
}


  void publishObjectState(const std::string & object_id, TrackedEntity & entity, const rclcpp::Time & stamp)
  {
    if (!entity.filter->isInitialized()) {
      return;
    }

    const auto & x = entity.filter->state();

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "world";
    pose_msg.header.stamp = stamp;
    pose_msg.pose.position.x = x.p.x();
    pose_msg.pose.position.y = x.p.y();
    pose_msg.pose.position.z = x.p.z();
    pose_msg.pose.orientation.x = x.q.x();
    pose_msg.pose.orientation.y = x.q.y();
    pose_msg.pose.orientation.z = x.q.z();
    pose_msg.pose.orientation.w = x.q.w();
    pose_pub_->publish(pose_msg);

    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header = pose_msg.header;

    if (entity.last_output_valid) {
      const double dt = (stamp - entity.last_output_stamp).seconds();
      if (dt > 1e-6) {
        const Eigen::Vector3d v = (x.p - entity.last_position) / dt;
        twist_msg.twist.linear.x = v.x();
        twist_msg.twist.linear.y = v.y();
        twist_msg.twist.linear.z = v.z();

        const Eigen::Quaterniond dq = entity.last_orientation.conjugate() * x.q;
        const Eigen::Vector3d dtheta = pose_evaluator::quatLog(dq);
        const Eigen::Vector3d w = dtheta / dt;

        twist_msg.twist.angular.x = w.x();
        twist_msg.twist.angular.y = w.y();
        twist_msg.twist.angular.z = w.z();
      }
    }

    twist_pub_->publish(twist_msg);

    entity.last_position = x.p;
    entity.last_orientation = x.q;
    entity.last_output_stamp = stamp;
    entity.last_output_valid = true;

    (void)object_id; // при желании можно добавить object_id в отдельный топик/namespace
  }

  void detectionsCallback(const pose_evaluator::msg::DetectedPoints::SharedPtr msg)
  {
    const auto stamp = msg->header.stamp;
    const auto camera_id = msg->camera_id;
    const CameraIntrinsics K = intrinsicsFromCameraInfo(msg->camera_info);

    std::vector<WorldPointObservation> world_obs;
    std::unordered_map<std::string, std::vector<ObjectPointObservation>> object_obs_map;

    for (const auto & p : msg->points) {
      if (p.coordinate_frame == "world") {
        WorldPointObservation obs;
        obs.point_world = Eigen::Vector3d(p.x, p.y, p.z);
        obs.pixel = Eigen::Vector2d(p.u, p.v);
        world_obs.push_back(obs);
      } else {
        // считаем, что coordinate_frame задает object id, например "object_1"
        ObjectPointObservation obs;
        obs.point_object = Eigen::Vector3d(p.x, p.y, p.z);
        obs.pixel = Eigen::Vector2d(p.u, p.v);
        object_obs_map[p.coordinate_frame].push_back(obs);
      }
    }

    // 1. update camera
    auto & camera_entity = getOrCreateCameraEntity(camera_id);
    updateCameraEntity(camera_entity, K, world_obs, stamp);

    if (!camera_entity.filter->isInitialized()) {
      return;
    }

    const State camera_state = camera_entity.filter->state();

    // 2. update all objects seen by this camera
    for (auto & kv : object_obs_map) {
      const std::string & object_id = kv.first;
      auto & observations = kv.second;

      auto & object_entity = getOrCreateObjectEntity(object_id);
      updateObjectEntity(object_entity, K, camera_state, observations, stamp);
      publishObjectState(object_id, object_entity, stamp);
    }
  }

private:
  std::string camera_filter_type_;
  std::string object_filter_type_;

  double sigma_a_{0.5};
  double sigma_alpha_{0.5};
  double sigma_px_{1.5};

  std::unordered_map<std::string, TrackedEntity> camera_entities_;
  std::unordered_map<std::string, TrackedEntity> object_entities_;

  rclcpp::Subscription<pose_evaluator::msg::DetectedPoints>::SharedPtr detections_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoseEvaluatorNode>());
  rclcpp::shutdown();
  return 0;
}
