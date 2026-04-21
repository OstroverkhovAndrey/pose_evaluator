#pragma once
#include "pose_evaluator/measurement_model.hpp"
#include <vector>

namespace pose_evaluator
{

struct CameraIntrinsics
{
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
};

struct WorldPointObservation
{
  Eigen::Vector3d point_world;
  Eigen::Vector2d pixel;
};

class CameraPinholeMeasurementModel : public IMeasurementModel
{
public:
  CameraPinholeMeasurementModel(
    const CameraIntrinsics & K,
    const std::vector<WorldPointObservation> & observations,
    double sigma_px)
  : K_(K), observations_(observations), sigma_px_(sigma_px) {}

  int measurementDim() const override
  {
    return static_cast<int>(2 * observations_.size());
  }

  Eigen::VectorXd predictMeasurement(const State & camera_state) const override
  {
    Eigen::VectorXd zhat(measurementDim());

    const Eigen::Matrix3d R_wc = camera_state.q.toRotationMatrix();
    const Eigen::Matrix3d R_cw = R_wc.transpose();
    const Eigen::Vector3d p_wc = camera_state.p;

    for (size_t i = 0; i < observations_.size(); ++i) {
      const Eigen::Vector3d Xw = observations_[i].point_world;
      const Eigen::Vector3d Xc = R_cw * (Xw + p_wc);

      const double u = K_.fx * (Xc.x() / Xc.z()) + K_.cx;
      const double v = K_.fy * (Xc.y() / Xc.z()) + K_.cy;

      zhat(2 * i + 0) = u;
      zhat(2 * i + 1) = v;
    }

    return zhat;
  }

  Eigen::MatrixXd measurementCov() const override
  {
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(measurementDim(), measurementDim());
    for (int i = 0; i < measurementDim(); ++i) {
      R(i, i) = sigma_px_ * sigma_px_;
    }
    return R;
  }

  Eigen::VectorXd measurementVector() const
  {
    Eigen::VectorXd z(measurementDim());
    for (size_t i = 0; i < observations_.size(); ++i) {
      z(2 * i + 0) = observations_[i].pixel.x();
      z(2 * i + 1) = observations_[i].pixel.y();
    }
    return z;
  }

  const std::vector<WorldPointObservation> & getObservations() const { return observations_; }
  const CameraIntrinsics & getIntrinsics() const { return K_; }

private:
  CameraIntrinsics K_;
  std::vector<WorldPointObservation> observations_;
  double sigma_px_;
};

}  // namespace pose_evaluator
