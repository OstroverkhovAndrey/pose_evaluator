
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <thread>
#include <functional>
#include <atomic>

struct CameraParams {
    int camera_id;
    int camera_width;
    int camera_height;
    int publish_frequency;
};

class CameraCapture
{
public:
    using FrameCallback = std::function<void(const cv::Mat&)>;

    CameraCapture(CameraParams camera_params, FrameCallback frame_callback)
        : running_(true), frame_callback_(frame_callback)
    {
        cap_.open(camera_params.camera_id);

        cap_.set(cv::CAP_PROP_FRAME_WIDTH, camera_params.camera_width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_params.camera_height);
        cap_.set(cv::CAP_PROP_FPS, camera_params.publish_frequency);
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

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

class CameraPublisher : public rclcpp::Node
{
public:
    CameraPublisher()
        : Node("latest_image_publisher_node")
    {

        declare_parameters();
        get_parameters();

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic_, 10);

        camera_ = std::make_unique<CameraCapture>(
            camera_params_,
            [this](const cv::Mat& frame){ this->publishFrame(frame); }
        );

        RCLCPP_INFO(this->get_logger(), "CameraPublisher started. Publishing to: %s", output_topic_.c_str());
    }

    ~CameraPublisher()
    {
        if (camera_)
            camera_->stop();
    }

private:

    void declare_parameters() {
        declare_parameter<int>("camera_width", 640);
        declare_parameter<int>("camera_height", 480);
        declare_parameter<int>("camera_id", 0);
        declare_parameter<int>("publish_frequency", 1);
        declare_parameter<std::string>("output_topic", "/output_image");
    }

    void get_parameters() {
        camera_params_.camera_id = get_parameter("camera_id").as_int();
        camera_params_.camera_width = get_parameter("camera_width").as_int();
        camera_params_.camera_height = get_parameter("camera_height").as_int();
        camera_params_.publish_frequency = get_parameter("publish_frequency").as_int();
        output_topic_ = this->get_parameter("output_topic").as_string();
    }

    void publishFrame(const cv::Mat& frame)
    {
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = this->now();
        publisher_->publish(*msg);
    }
    
    std::string output_topic_;
    CameraParams camera_params_;
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
