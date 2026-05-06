#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <deque>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "pose_evaluator/msg/detected_points.hpp"
#include "pose_evaluator/filter_factory.hpp"
#include "pose_evaluator/white_noise_rigid_body_model.hpp"
#include "pose_evaluator/camera_pinhole_measurement_model.hpp"
#include "pose_evaluator/object_pinhole_measurement_model.hpp"

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
using pose_evaluator::WhiteNoiseRigidBodyModel;

class PoseEvaluatorNode : public rclcpp::Node
{
public:
  PoseEvaluatorNode()
  : Node("pose_evaluator_node")
  {
    camera_filter_type_ = declare_parameter<std::string>("camera_filter_type", "ukf");
    object_filter_type_ = declare_parameter<std::string>("object_filter_type", "ukf");

    sigma_a_ = declare_parameter<double>("sigma_a", 1e-5);
    sigma_alpha_ = declare_parameter<double>("sigma_alpha", 1e-5);
    sigma_px_ = declare_parameter<double>("sigma_px", 1.0);

    history_size_ = declare_parameter<int>("history_size", 100);

    detections_sub_ = create_subscription<pose_evaluator::msg::DetectedPoints>(
      "detected_points",
      10,
      std::bind(&PoseEvaluatorNode::detectionsCallback, this, std::placeholders::_1));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("estimated_pose", 10);
    twist_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("estimated_twist", 10);
  }

private:
  struct FilterSnapshot
  {
    rclcpp::Time stamp;
    State state;
    Cov12 covariance;
  };

  struct CameraMeasurementRecord
  {
    rclcpp::Time stamp;
    CameraIntrinsics K;
    std::vector<WorldPointObservation> observations;
  };

  struct ObjectMeasurementRecord
  {
    rclcpp::Time stamp;
    CameraIntrinsics K;
    State camera_state_at_measurement;
    std::vector<ObjectPointObservation> observations;
  };

  template<typename MeasurementRecordT>
  struct TrackedEntity
  {
    std::unique_ptr<IFilter> filter;
    rclcpp::Time last_stamp{0, 0, RCL_ROS_TIME};

    // История состояний и измерений для поддержки отката
    std::deque<FilterSnapshot> snapshots;
    std::deque<MeasurementRecordT> measurements;
  };

  using CameraEntity = TrackedEntity<CameraMeasurementRecord>;
  using ObjectEntity = TrackedEntity<ObjectMeasurementRecord>;

  // ------------------------------------------------------------
  // Создание / получение фильтра камеры
  // ------------------------------------------------------------
  CameraEntity & getOrCreateCameraEntity(const std::string & camera_id)
  {
    auto it = camera_entities_.find(camera_id);
    if (it != camera_entities_.end()) {
      return it->second;
    }

    auto process_model = std::make_shared<WhiteNoiseRigidBodyModel>(sigma_a_, sigma_alpha_);

    CameraEntity entity;
    entity.filter = makeCameraFilter(camera_filter_type_, process_model);

    auto [new_it, inserted] = camera_entities_.emplace(camera_id, std::move(entity));
    (void)inserted;
    return new_it->second;
  }

  // ------------------------------------------------------------
  // Создание / получение фильтра объекта
  // ------------------------------------------------------------
  ObjectEntity & getOrCreateObjectEntity(const std::string & object_id)
  {
    auto it = object_entities_.find(object_id);
    if (it != object_entities_.end()) {
      return it->second;
    }

    auto process_model = std::make_shared<WhiteNoiseRigidBodyModel>(sigma_a_, sigma_alpha_);

    ObjectEntity entity;
    entity.filter = makeObjectFilter(object_filter_type_, process_model);

    auto [new_it, inserted] = object_entities_.emplace(object_id, std::move(entity));
    (void)inserted;
    return new_it->second;
  }

  // ------------------------------------------------------------
  // Внутренние параметры камеры из CameraInfo
  // ------------------------------------------------------------
  static CameraIntrinsics intrinsicsFromCameraInfo(const sensor_msgs::msg::CameraInfo & info)
  {
    CameraIntrinsics K;
    K.fx = info.k[0];
    K.fy = info.k[4];
    K.cx = info.k[2];
    K.cy = info.k[5];
    return K;
  }

  void trimSnapshots(std::deque<FilterSnapshot> & snapshots)
  {
    while (static_cast<int>(snapshots.size()) > history_size_) {
      snapshots.pop_front();
    }
  }

  template<typename T>
  void trimMeasurements(std::deque<T> & measurements)
  {
    while (static_cast<int>(measurements.size()) > history_size_) {
      measurements.pop_front();
    }
  }

  template<typename T>
  void insertMeasurementSorted(std::deque<T> & measurements, const T & m)
  {
    auto it = std::upper_bound(
      measurements.begin(), measurements.end(), m.stamp,
      [](const rclcpp::Time & t, const T & rec) { return t < rec.stamp; });

    measurements.insert(it, m);
  }

  // ------------------------------------------------------------
  // Инициализация фильтра камеры через solvePnP
  // solvePnP: X_c = R_cw X_w + t_cw
  // Фильтр хранит camera in world:
  //   R_wc = R_cw^T
  //   p_wc = -R_cw^T t_cw
  // ------------------------------------------------------------
  bool initializeCameraFilterFromPnP(
    CameraEntity & entity,
    const CameraIntrinsics & K,
    const std::vector<WorldPointObservation> & observations)
  {
    if (observations.size() < 4) {
      return false;
    }

    std::vector<cv::Point3f> world_points;
    std::vector<cv::Point2f> image_points;
    for (const auto & obs : observations) {
      world_points.emplace_back(obs.point_world.x(), obs.point_world.y(), obs.point_world.z());
      image_points.emplace_back(obs.pixel.x(), obs.pixel.y());
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

    Eigen::Matrix3d R_wc = R_cw.transpose();
    Eigen::Vector3d p_wc = -R_wc * t_cw;

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

  // ------------------------------------------------------------
  // Инициализация фильтра объекта через solvePnP
  // solvePnP: X_c = R_co X_o + t_co
  // Камера в мире:
  //   X_w = R_wc X_c + p_wc
  // Следовательно:
  //   R_wo = R_wc R_co
  //   p_wo = p_wc + R_wc t_co
  // ------------------------------------------------------------
  bool initializeObjectFilterFromPnP(
    ObjectEntity & entity,
    const CameraIntrinsics & K,
    const State & camera_state,
    const std::vector<ObjectPointObservation> & observations)
  {
    if (observations.size() < 4) {
      return false;
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    for (const auto & obs : observations) {
      object_points.emplace_back(obs.point_object.x(), obs.point_object.y(), obs.point_object.z());
      image_points.emplace_back(obs.pixel.x(), obs.pixel.y());
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

  void saveSnapshot(CameraEntity & entity, const rclcpp::Time & stamp)
  {
    if (!entity.filter->isInitialized()) {
      return;
    }
    entity.snapshots.push_back(FilterSnapshot{stamp, entity.filter->state(), entity.filter->covariance()});
    trimSnapshots(entity.snapshots);
  }

  void saveSnapshot(ObjectEntity & entity, const rclcpp::Time & stamp)
  {
    if (!entity.filter->isInitialized()) {
      return;
    }
    entity.snapshots.push_back(FilterSnapshot{stamp, entity.filter->state(), entity.filter->covariance()});
    trimSnapshots(entity.snapshots);
  }

  bool restoreSnapshot(CameraEntity & entity, const FilterSnapshot & snapshot)
  {
    entity.filter->initialize(snapshot.state, snapshot.covariance);
    entity.last_stamp = snapshot.stamp;
    return true;
  }

  bool restoreSnapshot(ObjectEntity & entity, const FilterSnapshot & snapshot)
  {
    entity.filter->initialize(snapshot.state, snapshot.covariance);
    entity.last_stamp = snapshot.stamp;
    return true;
  }

  // ------------------------------------------------------------
  // Применение одного измерения камеры
  // ------------------------------------------------------------
  void applyCameraMeasurement(CameraEntity & entity, const CameraMeasurementRecord & rec)
  {
    if (rec.observations.size() < 4) {
      return;
    }

    if (!entity.filter->isInitialized()) {
      if (!initializeCameraFilterFromPnP(entity, rec.K, rec.observations)) {
        return;
      }
      entity.last_stamp = rec.stamp;
      saveSnapshot(entity, rec.stamp);
      return;
    }

    double dt = (rec.stamp - entity.last_stamp).seconds();
    if (dt < 0.0) {
      dt = 0.0;
    }

    entity.filter->predict(dt);

    CameraPinholeMeasurementModel model(rec.K, rec.observations, sigma_px_);
    Eigen::VectorXd z = model.measurementVector();
    entity.filter->update(z, model);

    entity.last_stamp = rec.stamp;
    saveSnapshot(entity, rec.stamp);
  }

  // ------------------------------------------------------------
  // Применение одного измерения объекта
  // ------------------------------------------------------------
  void applyObjectMeasurement(ObjectEntity & entity, const ObjectMeasurementRecord & rec)
  {
    if (rec.observations.size() < 4) {
      return;
    }

    if (!entity.filter->isInitialized()) {
      if (!initializeObjectFilterFromPnP(entity, rec.K, rec.camera_state_at_measurement, rec.observations)) {
        return;
      }
      entity.last_stamp = rec.stamp;
      saveSnapshot(entity, rec.stamp);
      return;
    }

    double dt = (rec.stamp - entity.last_stamp).seconds();
    if (dt < 0.0) {
      dt = 0.0;
    }

    entity.filter->predict(dt);

    ObjectPinholeMeasurementModel model(rec.K, rec.camera_state_at_measurement, rec.observations, sigma_px_);
    Eigen::VectorXd z = model.measurementVector();
    entity.filter->update(z, model);

    entity.last_stamp = rec.stamp;
    saveSnapshot(entity, rec.stamp);
  }

  // ------------------------------------------------------------
  // Replay камеры при поступлении измерения из прошлого
  // ------------------------------------------------------------
  void replayCameraEntity(CameraEntity & entity, const rclcpp::Time & stamp)
  {
    if (entity.snapshots.empty()) {
      return;
    }

    auto snapshot_it = entity.snapshots.begin();
    for (auto it = entity.snapshots.begin(); it != entity.snapshots.end(); ++it) {
      if (it->stamp <= stamp) {
        snapshot_it = it;
      } else {
        break;
      }
    }

    const FilterSnapshot base_snapshot = *snapshot_it;
    restoreSnapshot(entity, base_snapshot);

    while (!entity.snapshots.empty() && entity.snapshots.back().stamp > base_snapshot.stamp) {
      entity.snapshots.pop_back();
    }

    for (const auto & rec : entity.measurements) {
      if (rec.stamp > base_snapshot.stamp) {
        applyCameraMeasurement(entity, rec);
      }
    }
  }

  // ------------------------------------------------------------
  // Replay объекта при поступлении измерения из прошлого
  // ------------------------------------------------------------
  void replayObjectEntity(ObjectEntity & entity, const rclcpp::Time & stamp)
  {
    if (entity.snapshots.empty()) {
      return;
    }

    auto snapshot_it = entity.snapshots.begin();
    for (auto it = entity.snapshots.begin(); it != entity.snapshots.end(); ++it) {
      if (it->stamp <= stamp) {
        snapshot_it = it;
      } else {
        break;
      }
    }

    const FilterSnapshot base_snapshot = *snapshot_it;
    restoreSnapshot(entity, base_snapshot);

    while (!entity.snapshots.empty() && entity.snapshots.back().stamp > base_snapshot.stamp) {
      entity.snapshots.pop_back();
    }

    for (const auto & rec : entity.measurements) {
      if (rec.stamp > base_snapshot.stamp) {
        applyObjectMeasurement(entity, rec);
      }
    }
  }

  // ------------------------------------------------------------
  // Публикация состояния объекта
  // Скорости публикуются напрямую из состояния фильтра:
  //   linear = x.v
  //   angular = x.w
  // ------------------------------------------------------------
  void publishObjectState(const std::string & object_id, ObjectEntity & entity, const rclcpp::Time & stamp)
  {
    if (!entity.filter->isInitialized()) {
      return;
    }

    const auto & x = entity.filter->state();

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "world";

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

    // Используем скорости прямо из состояния фильтра
    twist_msg.twist.linear.x = x.v.x();
    twist_msg.twist.linear.y = x.v.y();
    twist_msg.twist.linear.z = x.v.z();

    twist_msg.twist.angular.x = x.w.x();
    twist_msg.twist.angular.y = x.w.y();
    twist_msg.twist.angular.z = x.w.z();

    twist_pub_->publish(twist_msg);

    (void)object_id;
  }

  // ------------------------------------------------------------
  // Главный callback
  // 1. Разделение входных точек на world / object
  // 2. Обновление фильтра камеры
  // 3. Обновление фильтров объектов
  // ------------------------------------------------------------
  void detectionsCallback(const pose_evaluator::msg::DetectedPoints::SharedPtr msg)
  {
    const rclcpp::Time stamp = msg->header.stamp;
    const std::string camera_id = msg->camera_id;
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
        ObjectPointObservation obs;
        obs.point_object = Eigen::Vector3d(p.x, p.y, p.z);
        obs.pixel = Eigen::Vector2d(p.u, p.v);
        object_obs_map[p.coordinate_frame].push_back(obs);
      }
    }

    // ---------- Камера ----------
    auto & camera_entity = getOrCreateCameraEntity(camera_id);

    CameraMeasurementRecord camera_rec;
    camera_rec.stamp = stamp;
    camera_rec.K = K;
    camera_rec.observations = world_obs;

    insertMeasurementSorted(camera_entity.measurements, camera_rec);
    trimMeasurements(camera_entity.measurements);

    if (!camera_entity.filter->isInitialized() || stamp >= camera_entity.last_stamp) {
      applyCameraMeasurement(camera_entity, camera_rec);
    } else {
      replayCameraEntity(camera_entity, stamp);
    }

    if (!camera_entity.filter->isInitialized()) {
      return;
    }

    const State camera_state = camera_entity.filter->state();

    // ---------- Объекты ----------
    for (auto & kv : object_obs_map) {
      const std::string & object_id = kv.first;
      auto & object_entity = getOrCreateObjectEntity(object_id);

      ObjectMeasurementRecord object_rec;
      object_rec.stamp = stamp;
      object_rec.K = K;
      object_rec.camera_state_at_measurement = camera_state;
      object_rec.observations = kv.second;

      insertMeasurementSorted(object_entity.measurements, object_rec);
      trimMeasurements(object_entity.measurements);

      if (!object_entity.filter->isInitialized() || stamp >= object_entity.last_stamp) {
        applyObjectMeasurement(object_entity, object_rec);
      } else {
        replayObjectEntity(object_entity, stamp);
      }

      publishObjectState(object_id, object_entity, stamp);
    }
  }

private:
  std::string camera_filter_type_;
  std::string object_filter_type_;

  double sigma_a_;
  double sigma_alpha_;
  double sigma_px_;
  int history_size_;

  std::unordered_map<std::string, CameraEntity> camera_entities_;
  std::unordered_map<std::string, ObjectEntity> object_entities_;

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
