
// публикует изображения, в зависимости от режима работы может 
// публиковать иил изображение из файла или изображения потупающие 
// от видеокамеры

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include "pose_evaluator/msg/camera_image_stamped.hpp"

class CameraImagePublisher : public rclcpp::Node
{
public:
    CameraImagePublisher()
      : Node("image_publisher")
    {

        declare_parameters();
        get_parameters();

        publisher_ = create_publisher<pose_evaluator::msg::CameraImageStamped>(output_topic_, 10);

        cam_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "camera", camera_info_path_);

        RCLCPP_INFO(get_logger(), mode_.c_str());
        if (mode_ == "camera") {
            cap_ = std::make_unique<cv::VideoCapture>(camera_id_);
            if (!cap_->isOpened()) {
                RCLCPP_ERROR(get_logger(), "Cannot open camera %d", camera_id_);
                throw std::runtime_error("Camera open failed");
            }
            cap_->set(cv::CAP_PROP_FRAME_WIDTH, camera_width_);
            cap_->set(cv::CAP_PROP_FRAME_HEIGHT, camera_height_);
            cap_->set(cv::CAP_PROP_FPS, publish_frequency_);
            cap_->set(cv::CAP_PROP_FOURCC,
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);
        } else if (mode_ == "file") {
            if (debug_image_path_.empty()) {
                RCLCPP_ERROR(get_logger(), "debug_image_path parameter is empty!");
                throw std::runtime_error("debug_image_path empty");
            }
            cv::Mat img = cv::imread(debug_image_path_);
            if (img.empty()) {
                RCLCPP_ERROR(get_logger(), "Cannot load image %s", debug_image_path_.c_str());
                throw std::runtime_error("Cannot load debug image");
            }
            debug_image_ = img;
        } else {
            RCLCPP_ERROR(get_logger(), "Unknown mode %s (use 'camera' or 'file')", mode_.c_str());
            throw std::runtime_error("Unknown mode");
        }

        timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0/publish_frequency_),
            std::bind(&CameraImagePublisher::timer_callback, this)
        );

        RCLCPP_INFO(get_logger(), "CameraImagePublisher started, mode: %s.", mode_.c_str());
    }

    ~CameraImagePublisher() {
        if (cap_) { cap_->release(); }
    }

private:

    void declare_parameters() {
        declare_parameter<std::string>("mode", "camera");
        declare_parameter<std::string>("camera_info_path", "");
        declare_parameter<std::string>("debug_image_path", "");
        declare_parameter<int>("camera_width", 640);
        declare_parameter<int>("camera_height", 480);
        declare_parameter<int>("camera_id", 0);
        declare_parameter<int>("publish_frequency", 1);
        declare_parameter<std::string>("output_topic", "/output_image");
    }

    void get_parameters() {
        mode_ = get_parameter("mode").as_string();
        camera_info_path_ = get_parameter("camera_info_path").as_string();
        debug_image_path_ = get_parameter("debug_image_path").as_string();
        camera_width_ = get_parameter("camera_width").as_int();
        camera_height_ = get_parameter("camera_height").as_int();
        camera_id_ = get_parameter("camera_id").as_int();
        publish_frequency_ = get_parameter("publish_frequency").as_int();
        output_topic_ = this->get_parameter("output_topic").as_string();
    }

    void timer_callback() {
        pose_evaluator::msg::CameraImageStamped msg;
        msg.header.stamp = now();
        msg.header.frame_id = "camera";
        msg.camera_info = cam_info_manager_->getCameraInfo();

        cv::Mat frame;
        if (mode_ == "camera") {
            if (!cap_->read(frame)) {
                RCLCPP_WARN(get_logger(), "Failed to get frame from camera");
                return;
            }
        } else if (mode_ == "file") {
            frame = debug_image_;
        }
        // To ROS msg
        auto image_msg = cv_bridge::CvImage(
            msg.header, "bgr8", frame
        ).toImageMsg();
        msg.image = *image_msg;

        publisher_->publish(msg);
    }

    std::string output_topic_;
    std::string mode_, camera_info_path_, debug_image_path_;
    int camera_width_, camera_height_, camera_id_;
    int publish_frequency_;

    std::unique_ptr<cv::VideoCapture> cap_;
    cv::Mat debug_image_;
    rclcpp::Publisher<pose_evaluator::msg::CameraImageStamped>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cam_info_manager_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraImagePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
