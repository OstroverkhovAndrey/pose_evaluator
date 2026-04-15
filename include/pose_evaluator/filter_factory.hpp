#pragma once

#include "pose_evaluator/filter.hpp"
#include "pose_evaluator/ukf.hpp"
#include "pose_evaluator/simple_filter.hpp"
#include "pose_evaluator/process_model.hpp"

#include <memory>
#include <string>
#include <stdexcept>

namespace pose_evaluator
{

inline std::unique_ptr<IFilter> makeCameraFilter(
  const std::string & type,
  std::shared_ptr<IProcessModel> process_model)
{
  if (type == "ukf") {
    return std::make_unique<UnscentedKalmanFilter>(process_model);
  }

  if (type == "simple") {
    return std::make_unique<SimpleFilter>(SimpleFilter::Mode::CameraInWorld);
  }

  throw std::runtime_error("Unsupported camera filter type: " + type);
}

inline std::unique_ptr<IFilter> makeObjectFilter(
  const std::string & type,
  std::shared_ptr<IProcessModel> process_model)
{
  if (type == "ukf") {
    return std::make_unique<UnscentedKalmanFilter>(process_model);
  }

  if (type == "simple") {
    return std::make_unique<SimpleFilter>(SimpleFilter::Mode::ObjectInWorld);
  }

  throw std::runtime_error("Unsupported object filter type: " + type);
}

}  // namespace pose_evaluator
