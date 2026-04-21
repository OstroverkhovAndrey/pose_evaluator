#pragma once
#include "pose_evaluator/measurement_model.hpp"
#include "pose_evaluator/camera_pinhole_measurement_model.hpp"
#include <vector>

namespace pose_evaluator
{

struct ObjectPointObservation
{
  Eigen::Vector3d point_object;
  Eigen::Vector2d pixel;
};

class ObjectPinholeMeasurementModel : public IMeasurementModel
{
public:
  ObjectPinholeMeasurementModel(
    const CameraIntrinsics & K,
    const State & camera_state,
    const std::vector<ObjectPointObservation> & observations,
    double sigma_px)
  : K_(K), camera_state_(camera_state), observations_(observations), sigma_px_(sigma_px) {}

  int measurementDim() const override
  {
    return static_cast<int>(2 * observations_.size());
  }

  Eigen::VectorXd predictMeasurement(const State & object_state) const override
  {
    Eigen::VectorXd zhat(measurementDim());

    const Eigen::Matrix3d R_wc = camera_state_.q.toRotationMatrix();
    const Eigen::Matrix3d R_cw = R_wc.transpose();
    const Eigen::Vector3d p_wc = camera_state_.p;

    const Eigen::Matrix3d R_wo = object_state.q.toRotationMatrix();
    const Eigen::Vector3d p_wo = object_state.p;

    for (size_t i = 0; i < observations_.size(); ++i) {
      const Eigen::Vector3d Xo = observations_[i].point_object;
      const Eigen::Vector3d Xw = R_wo * Xo + p_wo;
      const Eigen::Vector3d Xc = R_cw * (Xw + p_wc);

        if (!std::isfinite(Xc.z()) || std::abs(Xc.z()) < 1e-9) {
          std::cout << "Bad depth Xc.z = " << Xc.z() << std::endl;
        }

        if (Xc.z() <= 1e-6) {
          std::cout << "Point behind camera or too close: " << Xc.transpose() << std::endl;
        }


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

  const std::vector<ObjectPointObservation> & getObservations() const { return observations_; }
  const CameraIntrinsics & getIntrinsics() const { return K_; }
  const State & getCameraState() const { return camera_state_; }

private:
  CameraIntrinsics K_;
  State camera_state_;
  std::vector<ObjectPointObservation> observations_;
  double sigma_px_;
};

}  // namespace pose_evaluator
