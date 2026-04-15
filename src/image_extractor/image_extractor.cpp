
// Получает изображение типа CameraImageStamped извлекает
// из него Image и публикует его

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "pose_evaluator/msg/camera_image_stamped.hpp"

class CameraImageExtractor : public rclcpp::Node
{
public:
  CameraImageExtractor()
    : Node("image_extractor")
  {
    this->declare_parameter<std::string>("input_topic", "/camera_image");
    this->declare_parameter<std::string>("output_topic", "/image");

    std::string input_topic = this->get_parameter("input_topic").as_string();
    std::string output_topic = this->get_parameter("output_topic").as_string();

    pub_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic, 10);
    sub_ = this->create_subscription<pose_evaluator::msg::CameraImageStamped>(
      input_topic, 10,
      [this](const pose_evaluator::msg::CameraImageStamped::SharedPtr msg)
      {
        pub_->publish(msg->image);
      });

    RCLCPP_INFO(this->get_logger(), "Started. Subscribing: %s; Publishing: %s",
                input_topic.c_str(), output_topic.c_str());
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::Subscription<pose_evaluator::msg::CameraImageStamped>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraImageExtractor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
