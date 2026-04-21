
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <thread>
#include <functional>
#include <atomic>

class CameraCapture
{
public:
    using FrameCallback = std::function<void(const cv::Mat&)>;

    CameraCapture(int camera_index, FrameCallback frame_callback)
        : running_(true), frame_callback_(frame_callback)
    {
        cap_.open(camera_index);
        if (!cap_.isOpened())
            throw std::runtime_error("Cannot open camera");
        capture_thread_ = std::thread(&CameraCapture::captureLoop, this);
    }
    ~CameraCapture()
    {
        stop();
    }
    void stop()
    {
        running_ = false;
        if (capture_thread_.joinable())
            capture_thread_.join();
        cap_.release();
    }
private:
    // TODO надо брать последний кадр 
    void captureLoop()
    {
        cv::Mat frame;
        while (running_)
        {
            if (cap_.read(frame))
            {
                if (frame_callback_)
                    frame_callback_(frame);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    cv::VideoCapture cap_;
    std::thread capture_thread_;
    std::atomic<bool> running_;
    FrameCallback frame_callback_;
};

// ROS2-нода публикации кадров по готовности
class CameraPublisher : public rclcpp::Node
{
public:
    CameraPublisher()
        : Node("camera_publisher_node")
    {
        this->declare_parameter("camera_index", 0);
        this->declare_parameter("publish_topic", "/camera/image_raw");

        int camera_index = this->get_parameter("camera_index").as_int();
        std::string publish_topic = this->get_parameter("publish_topic").as_string();

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(publish_topic, 10);

        // ВАЖНО: захватчик камеры теперь создаётся с LAMBDA callback!
        camera_ = std::make_unique<CameraCapture>(
            camera_index,
            [this](const cv::Mat& frame){ this->publishFrame(frame); }
        );

        RCLCPP_INFO(this->get_logger(), "CameraPublisher started. Publishing to: %s", publish_topic.c_str());
    }

    ~CameraPublisher()
    {
        if (camera_)
            camera_->stop();
    }

private:
    void publishFrame(const cv::Mat& frame)
    {
        // Публикация только, если есть подписчики (можно проверить, но не обязательно)
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = this->now();
        publisher_->publish(*msg);
    }

    std::unique_ptr<CameraCapture> camera_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
