#pragma once

#include "pose_evaluator/measurement_model.hpp"

namespace pose_evaluator
{

class PlaneMotionMeasurementModel : public IMeasurementModel
{
public:
  PlaneMotionMeasurementModel(
    double sigma_pz,
    double sigma_vz,
    double sigma_nx,
    double sigma_ny,
    double sigma_wx,
    double sigma_wy)
  {
    R_ = Eigen::MatrixXd::Zero(6, 6);
    R_(0, 0) = sigma_pz * sigma_pz;
    R_(1, 1) = sigma_vz * sigma_vz;
    R_(2, 2) = sigma_nx * sigma_nx;
    R_(3, 3) = sigma_ny * sigma_ny;
    R_(4, 4) = sigma_wx * sigma_wx;
    R_(5, 5) = sigma_wy * sigma_wy;
  }

  int measurementDim() const override
  {
    return 6;
  }

  Eigen::VectorXd predictMeasurement(const State & x) const override
  {
    Eigen::VectorXd zhat(6);

    // Ограничение высоты и вертикальной скорости
    const double pz = x.p.z();
    const double vz = x.v.z();

    // n = R(q) * e_z — направление локальной оси z тела в мировой системе
    const Eigen::Vector3d ez(0.0, 0.0, 1.0);
    const Eigen::Vector3d n = x.q.toRotationMatrix() * ez;

    // Для плоского движения хотим n_x = 0, n_y = 0
    const double nx = n.x();
    const double ny = n.y();

    // Ограничение угловой скорости вокруг x и y
    const double wx = x.w.x();
    const double wy = x.w.y();

    zhat << pz, vz, nx, ny, wx, wy;
    return zhat;
  }

  Eigen::MatrixXd measurementCov() const override
  {
    return R_;
  }

  Eigen::VectorXd measurementVector() const
  {
    // Псевдоизмерение: все ограничиваемые компоненты равны нулю
    return Eigen::VectorXd::Zero(6);
  }

private:
  Eigen::MatrixXd R_;
};

}  // namespace pose_evaluator
