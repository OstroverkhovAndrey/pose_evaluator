#pragma once

#include "pose_evaluator/filter.hpp"
#include "pose_evaluator/camera_pinhole_measurement_model.hpp"
#include "pose_evaluator/object_pinhole_measurement_model.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace pose_evaluator
{

class SimpleFilter : public IFilter
{
public:
  enum class Mode
  {
    CameraInWorld,   // filter estimates T_wc using world points
    ObjectInWorld    // filter estimates T_wo using object points + current camera pose
  };

  explicit SimpleFilter(Mode mode)
  : mode_(mode)
  {}

  void initialize(const State & x0, const Cov12 & P0) override
  {
    x_ = x0;
    P_ = P0;
    initialized_ = true;
  }

  bool isInitialized() const override
  {
    return initialized_;
  }

  void predict(double /*dt*/) override
  {
    // no prediction
  }

  void update(const Eigen::VectorXd &, const IMeasurementModel & model) override
  {
    switch (mode_) {
      case Mode::CameraInWorld:
        updateCamera(model);
        break;
      case Mode::ObjectInWorld:
        updateObject(model);
        break;
    }
  }

  const State & state() const override { return x_; }
  const Cov12 & covariance() const override { return P_; }

private:
  void updateCamera(const IMeasurementModel & model)
  {
    auto const * meas = dynamic_cast<const CameraPinholeMeasurementModel *>(&model);
    if (!meas) {
      return;
    }

    const auto & obs = meas->getObservations();
    const auto & K = meas->getIntrinsics();

    if (obs.size() < 4) {
      return;
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(obs.size());
    image_points.reserve(obs.size());

    for (const auto & o : obs) {
      object_points.emplace_back(
        static_cast<float>(o.point_world.x()),
        static_cast<float>(o.point_world.y()),
        static_cast<float>(o.point_world.z()));
      image_points.emplace_back(
        static_cast<float>(o.pixel.x()),
        static_cast<float>(o.pixel.y()));
    }

    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
      K.fx, 0.0, K.cx,
      0.0, K.fy, K.cy,
      0.0, 0.0, 1.0);

    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat rvec, tvec;

    const bool ok = cv::solvePnP(object_points, image_points, camera_matrix, dist, rvec, tvec);
    if (!ok) {
      return;
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

    // solvePnP gives world in camera: Xc = R_cw Xw + t_cw
    // We want camera in world:
    const Eigen::Matrix3d R_wc = R_cw.transpose();
    const Eigen::Vector3d p_wc = -R_wc * t_cw;

    x_.q = Eigen::Quaterniond(R_wc).normalized();
    x_.p = p_wc;
    x_.v.setZero();
    x_.w.setZero();
    initialized_ = true;
  }

  void updateObject(const IMeasurementModel & model)
  {
    auto const * meas = dynamic_cast<const ObjectPinholeMeasurementModel *>(&model);
    if (!meas) {
      return;
    }

    const auto & obs = meas->getObservations();
    const auto & K = meas->getIntrinsics();
    const auto & camera_state = meas->getCameraState();

    if (obs.size() < 4) {
      return;
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(obs.size());
    image_points.reserve(obs.size());

    for (const auto & o : obs) {
      object_points.emplace_back(
        static_cast<float>(o.point_object.x()),
        static_cast<float>(o.point_object.y()),
        static_cast<float>(o.point_object.z()));
      image_points.emplace_back(
        static_cast<float>(o.pixel.x()),
        static_cast<float>(o.pixel.y()));
    }

    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
      K.fx, 0.0, K.cx,
      0.0, K.fy, K.cy,
      0.0, 0.0, 1.0);

    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat rvec, tvec;

    const bool ok = cv::solvePnP(object_points, image_points, camera_matrix, dist, rvec, tvec);
    if (!ok) {
      return;
    }

    cv::Mat Rco_cv;
    cv::Rodrigues(rvec, Rco_cv);

    Eigen::Matrix3d R_co;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        R_co(r, c) = Rco_cv.at<double>(r, c);
      }
    }

    const Eigen::Vector3d t_co(
      tvec.at<double>(0, 0),
      tvec.at<double>(1, 0),
      tvec.at<double>(2, 0));

    // solvePnP gives object in camera:
    // Xc = R_co Xo + t_co
    //
    // camera filter stores camera in world:
    // Xw = R_wc Xc + p_wc
    //
    // Therefore object in world:
    // R_wo = R_wc R_co
    // p_wo = R_wc t_co + p_wc

    const Eigen::Matrix3d R_wc = camera_state.q.toRotationMatrix();
    const Eigen::Matrix3d R_wo = R_wc * R_co;
    const Eigen::Vector3d p_wo = R_wc * t_co + camera_state.p;

    x_.q = Eigen::Quaterniond(R_wo).normalized();
    x_.p = p_wo;
    x_.v.setZero();
    x_.w.setZero();
    initialized_ = true;
  }

private:
  Mode mode_;
  State x_;
  Cov12 P_ = Cov12::Identity();
  bool initialized_{false};
};

}  // namespace pose_evaluator
