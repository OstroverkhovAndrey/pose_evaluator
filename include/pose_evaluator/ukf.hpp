#pragma once
#include "pose_evaluator/filter.hpp"
#include "pose_evaluator/process_model.hpp"
#include "pose_evaluator/state_ops.hpp"

#include <memory>
#include <vector>
#include <stdexcept>

namespace pose_evaluator
{

class UnscentedKalmanFilter : public IFilter
{
public:
  struct Params
  {
    double alpha{1e-2};
    double beta{2.0};
    double kappa{0.0};
    int mean_iterations{6};
    double jitter{1e-9};
  };

  explicit UnscentedKalmanFilter(std::shared_ptr<IProcessModel> process_model)
  : UnscentedKalmanFilter(std::move(process_model), Params()) {}

  UnscentedKalmanFilter(
    std::shared_ptr<IProcessModel> process_model,
    const Params & params)
  : process_model_(std::move(process_model)), params_(params)
  {
    if (!process_model_) {
      throw std::runtime_error("process_model is null");
    }
  }

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

  void predict(double dt) override
  {
    if (!initialized_) {
      return;
    }

    constexpr int NX = 12;
    constexpr int NW = 6;
    constexpr int NA = NX + NW;

    const double lambda = params_.alpha * params_.alpha * (NA + params_.kappa) - NA;

    Eigen::Matrix<double, NA, NA> P_aug = Eigen::Matrix<double, NA, NA>::Zero();
    P_aug.block<12,12>(0,0) = P_;
    P_aug.block<6,6>(12,12) = process_model_->noiseCov(dt);
    P_aug += params_.jitter * Eigen::Matrix<double, NA, NA>::Identity();

    Eigen::Matrix<double, NA, NA> L = ((NA + lambda) * P_aug).llt().matrixL();

    std::vector<double> Wm(2 * NA + 1), Wc(2 * NA + 1);
    computeWeights(NA, lambda, Wm, Wc);

    std::vector<State> sigma_pred(2 * NA + 1);

    auto propagateSigma = [&](const Eigen::Matrix<double, NA, 1> & delta) {
      ErrorVec dx = delta.template segment<12>(0);
      ProcessNoiseVec dw = delta.template segment<6>(12);
      State sigma_state = StateOps::boxPlus(x_, dx);
      return process_model_->propagate(sigma_state, dw, dt);
    };

    sigma_pred[0] = propagateSigma(Eigen::Matrix<double, NA, 1>::Zero());

    for (int i = 0; i < NA; ++i) {
      sigma_pred[i + 1] = propagateSigma(L.col(i));
      sigma_pred[i + 1 + NA] = propagateSigma(-L.col(i));
    }

    State x_mean = computeStateMean(sigma_pred, Wm);

    Cov12 P = Cov12::Zero();
    for (int i = 0; i < 2 * NA + 1; ++i) {
      ErrorVec dx = StateOps::boxMinus(sigma_pred[i], x_mean);
      P += Wc[i] * dx * dx.transpose();
    }

    x_ = x_mean;
    P_ = 0.5 * (P + P.transpose());
  }

  void update(const Eigen::VectorXd & z, const IMeasurementModel & model) override
  {
    if (!initialized_) {
      return;
    }

    constexpr int NX = 12;
    constexpr int NA = NX;
    const int nz = model.measurementDim();

    const double lambda = params_.alpha * params_.alpha * (NA + params_.kappa) - NA;

    Eigen::Matrix<double, NX, NX> Pj = P_;
    Pj += params_.jitter * Eigen::Matrix<double, NX, NX>::Identity();
    Eigen::Matrix<double, NX, NX> L = ((NA + lambda) * Pj).llt().matrixL();

    std::vector<double> Wm(2 * NA + 1), Wc(2 * NA + 1);
    computeWeights(NA, lambda, Wm, Wc);

    std::vector<State> sigma_x(2 * NA + 1);
    sigma_x[0] = x_;

    for (int i = 0; i < NA; ++i) {
      sigma_x[i + 1] = StateOps::boxPlus(x_, L.col(i));
      sigma_x[i + 1 + NA] = StateOps::boxPlus(x_, -L.col(i));
    }

    std::vector<Eigen::VectorXd> sigma_z(2 * NA + 1, Eigen::VectorXd(nz));
    for (int i = 0; i < 2 * NA + 1; ++i) {
      sigma_z[i] = model.predictMeasurement(sigma_x[i]);
    }

    Eigen::VectorXd z_mean = Eigen::VectorXd::Zero(nz);
    for (int i = 0; i < 2 * NA + 1; ++i) {
      z_mean += Wm[i] * sigma_z[i];
    }

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nz, nz);
    Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(NX, nz);

    for (int i = 0; i < 2 * NA + 1; ++i) {
      ErrorVec dx = StateOps::boxMinus(sigma_x[i], x_);
      Eigen::VectorXd dz = sigma_z[i] - z_mean;
      S += Wc[i] * dz * dz.transpose();
      Pxz += Wc[i] * dx * dz.transpose();
    }

    S += model.measurementCov();

    Eigen::MatrixXd K = Pxz * S.ldlt().solve(Eigen::MatrixXd::Identity(nz, nz));
    ErrorVec dx = K * (z - z_mean);

    x_ = StateOps::boxPlus(x_, dx);
    P_ = P_ - K * S * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
  }

  const State & state() const override { return x_; }
  const Cov12 & covariance() const override { return P_; }

private:
  void computeWeights(
    int n,
    double lambda,
    std::vector<double> & Wm,
    std::vector<double> & Wc) const
  {
    Wm[0] = lambda / (n + lambda);
    Wc[0] = Wm[0] + (1.0 - params_.alpha * params_.alpha + params_.beta);

    for (int i = 1; i < 2 * n + 1; ++i) {
      Wm[i] = 1.0 / (2.0 * (n + lambda));
      Wc[i] = Wm[i];
    }
  }

  State computeStateMean(
    const std::vector<State> & sigma,
    const std::vector<double> & Wm) const
  {
    State mean = sigma[0];

    for (int iter = 0; iter < params_.mean_iterations; ++iter) {
      ErrorVec acc = ErrorVec::Zero();
      for (size_t i = 0; i < sigma.size(); ++i) {
        acc += Wm[i] * StateOps::boxMinus(sigma[i], mean);
      }
      mean = StateOps::boxPlus(mean, acc);
      if (acc.norm() < 1e-12) {
        break;
      }
    }

    return mean;
  }

private:
  std::shared_ptr<IProcessModel> process_model_;
  Params params_;

  State x_;
  Cov12 P_ = Cov12::Identity();
  bool initialized_{false};
};

}  // namespace pose_evaluator
