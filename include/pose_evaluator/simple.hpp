
#pragma once
#include "pose_evaluator/filter.hpp"
#include "pose_evaluator/process_model.hpp"
#include "pose_evaluator/state_ops.hpp"
#include "pose_evaluator/pinhole_point_measurement_model.hpp"

#include <memory>
#include <vector>
#include <stdexcept>

namespace pose_evaluator
{

class SimpleFilter : public IFilter
{
public:
  SimpleFilter() = default;

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
    // intentionally do nothing
  }

  void update(const Eigen::VectorXd &, const IMeasurementModel & model) override
  {
    auto const * meas = dynamic_cast<const PinholePointMeasurementModel *>(&model);
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
        static_cast<float>(o.point_body.x()),
        static_cast<float>(o.point_body.y()),
        static_cast<float>(o.point_body.z()));
      image_points.emplace_back(
        static_cast<float>(o.pixel.x()),
        static_cast<float>(o.pixel.y()));
    }

    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
      K.fx, 0.0, K.cx,
      0.0, K.fy, K.cy,
      0.0, 0.0, 1.0);

    // Предполагаем, что изображение уже rectified либо дисторсию игнорируем
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);

    cv::Mat rvec, tvec;
    bool ok = cv::solvePnP(object_points, image_points, camera_matrix, dist, rvec, tvec);
    if (!ok) {
      return;
    }

    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);

    Eigen::Matrix3d R_cb;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        R_cb(r, c) = Rcv.at<double>(r, c);
      }
    }

    x_.q = Eigen::Quaterniond(R_cb).normalized();
    x_.p = Eigen::Vector3d(
      tvec.at<double>(0, 0),
      tvec.at<double>(1, 0),
      tvec.at<double>(2, 0));
    x_.v.setZero();
    x_.w.setZero();
  }

  const State & state() const override { return x_; }
  const Cov12 & covariance() const override { return P_; }

private:
  State x_;
  Cov12 P_ = Cov12::Identity();
  bool initialized_{false};
};

}  // namespace pose_evaluator