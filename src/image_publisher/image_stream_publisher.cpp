#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include "pose_evaluator/msg/camera_image_stamped.hpp"

namespace fs = std::filesystem;

class ImageStreamPublisher : public rclcpp::Node {
public:
    ImageStreamPublisher()
      : Node("image_stream_publisher"), img_idx_(0)
    {
        declare_parameter<std::string>("img_dir", "./images");
        declare_parameter<std::string>("camera_info_path", "./camera_info.yaml");
        declare_parameter<double>("rate", 10.0);
        declare_parameter<std::string>("output_topic", "/output_image");

        img_dir_ = this->get_parameter("img_dir").as_string();
        camera_info_path_ = this->get_parameter("camera_info_path").as_string();
        pub_rate_ = this->get_parameter("rate").as_double();
        output_topic_ = this->get_parameter("output_topic").as_string();

        publisher_ = create_publisher<pose_evaluator::msg::CameraImageStamped>(output_topic_, 10); 

        cam_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "camera", camera_info_path_);

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / pub_rate_),
            std::bind(&ImageStreamPublisher::timer_callback, this)
        );
    }

private:

    void timer_callback() {
        if (img_idx_ > 1500) {
            // magic number
            return;
        }

        std::string img_path = img_dir_ + "/img_" + std::to_string(img_idx_) + ".jpg";
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();

        pose_evaluator::msg::CameraImageStamped msg;
        msg.header.stamp = now();
        msg.header.frame_id = "camera";
        msg.camera_info = cam_info_manager_->getCameraInfo();
        msg.camera_info.header = msg.header;
        msg.image = *img_msg;

        publisher_->publish(msg);
        img_idx_++;
    }

    std::string img_dir_, camera_info_path_;
    int img_idx_;
    double pub_rate_;
    std::string output_topic_;
    rclcpp::Publisher<pose_evaluator::msg::CameraImageStamped>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cam_info_manager_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageStreamPublisher>());
    rclcpp::shutdown();
    return 0;
}
