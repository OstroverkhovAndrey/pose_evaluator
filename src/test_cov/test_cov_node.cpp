
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <thread>
#include <functional>
#include <atomic>
#include <camera_info_manager/camera_info_manager.hpp>

#include "pose_evaluator/msg/detected_point.hpp"
#include "pose_evaluator/msg/detected_points.hpp"
#include "pose_evaluator/msg/camera_image_stamped.hpp"

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
        : Node("test_cov_node")
    {

        declare_parameters();
        get_parameters();
        dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

        cam_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "camera", camera_info_path_);

        camera_ = std::make_unique<CameraCapture>(
            camera_params_,
            [this](const cv::Mat& frame){ this->publishFrame(frame); }
        );
    }

    ~CameraPublisher()
    {
        if (camera_)
            camera_->stop();
    }

private:

    void declare_parameters() {
        declare_parameter<int>("camera_width", 1280);
        declare_parameter<int>("camera_height", 720);
        declare_parameter<int>("camera_id", 0);
        declare_parameter<int>("publish_frequency", 10);
    }

    void get_parameters() {
        camera_params_.camera_id = get_parameter("camera_id").as_int();
        camera_params_.camera_width = get_parameter("camera_width").as_int();
        camera_params_.camera_height = get_parameter("camera_height").as_int();
        camera_params_.publish_frequency = get_parameter("publish_frequency").as_int();
    }

    void publishFrame(const cv::Mat& image)
    {
        const auto cur_now = now();
        cv::Mat imageCopy;
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        cv::aruco::detectMarkers(image, dictionary_, corners, ids);

        if (ids.size() == 1 && ids[0] == 8) {
            for (int i = 0; i < 8; i+=2) {
                v[i].push_back(corners[0][i].x);
                v[i+1].push_back(corners[0][i+1].y);
            }
        }
        if (v[0].size() == 1000) {
            for (int i = 0; i < 8; ++i) {
                std::cout << "[";
                for (int j = 0; j < 1000; ++j) {
                    std::cout << v[i][j] << ", ";
                }
                std::cout << "]\n";
            }
        }
    }

    std::vector<std::vector<float>> v{8, std::vector<float>()};
    
    CameraParams camera_params_;
    std::unique_ptr<CameraCapture> camera_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cam_info_manager_;

    std::string camera_info_path_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
