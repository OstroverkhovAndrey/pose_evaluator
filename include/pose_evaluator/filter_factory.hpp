
#pragma once
#include "pose_evaluator/filter.hpp"
#include "pose_evaluator/ukf.hpp"
#include "pose_evaluator/simple.hpp"
#include <memory>
#include <string>
#include <stdexcept>

namespace pose_evaluator
{

inline std::unique_ptr<IFilter> makeFilter(
  const std::string & type,
  std::shared_ptr<IProcessModel> process_model)
{
  if (type == "ukf") {
    return std::make_unique<UnscentedKalmanFilter>(process_model);
  }
  if (type == "simple") {
    return std::make_unique<SimpleFilter>();
  }

  throw std::runtime_error("Unsupported filter type: " + type);
}

}  // namespace pose_evaluator
