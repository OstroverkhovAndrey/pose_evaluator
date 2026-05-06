
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
        : Node("latest_image_publisher_node")
    {

        declare_parameters();
        get_parameters();
        dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

        image_publisher_ = this->create_publisher<pose_evaluator::msg::CameraImageStamped>(output_image_topic_, 10);
        image_with_detected_markers_publisher_ = this->create_publisher<pose_evaluator::msg::CameraImageStamped>(output_image_with_detect_markers_topic_, 10);
        markers_publisher_ = this->create_publisher<pose_evaluator::msg::DetectedPoints>(
                markers_topic_, 10);
        cam_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "camera", camera_info_path_);

        camera_ = std::make_unique<CameraCapture>(
            camera_params_,
            [this](const cv::Mat& frame){ this->publishFrame(frame); }
        );

        RCLCPP_INFO(this->get_logger(), "CameraPublisher started. Publishing to: %s", output_image_topic_.c_str());
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

        declare_parameter<std::string>("output_image_topic", "/image");
        declare_parameter<std::string>("output_image_with_detect_markers_topic", "/image_with_detect_markers");
        declare_parameter<std::string>("markers_topic", "/coordinates_markers_in_image");
        declare_parameter<std::string>("camera_info_path", "");

        this->declare_parameter("world_name", "world");
        this->declare_parameter("count_world_marker", 6);
        for (int i = 0; i < this->get_parameter("count_world_marker").as_int(); ++i) {
          this->declare_parameter("id_world_marker_" + std::to_string(i), i);
          this->declare_parameter(
              "coord_world_marker_" + std::to_string(i),
              std::vector<double>{1, 2, 0, 2, 2, 0, 2, 1, 0, 1, 1, 0});
        }

        this->declare_parameter("count_object", 3);
        for (int i = 0; i < this->get_parameter("count_object").as_int(); ++i) {
          this->declare_parameter("count_object_" + std::to_string(i) + "_marker", 2);
          for (int j = 0;
               j < this->get_parameter("count_object_" + std::to_string(i) + "_marker")
                       .as_int();
               ++j) {
            this->declare_parameter(
                "id_object_" + std::to_string(i) + "_marker_" + std::to_string(j), i);
            this->declare_parameter(
                "coord_object_" + std::to_string(i) + "_marker_" + std::to_string(j),
                std::vector<double>{1, 2, 0, 2, 2, 0, 2, 1, 0, 1, 1, 0});
          }
        }
    }

    void get_parameters() {
        camera_params_.camera_id = get_parameter("camera_id").as_int();
        camera_params_.camera_width = get_parameter("camera_width").as_int();
        camera_params_.camera_height = get_parameter("camera_height").as_int();
        camera_params_.publish_frequency = get_parameter("publish_frequency").as_int();

        output_image_topic_ = get_parameter("output_image_topic").as_string();
        output_image_with_detect_markers_topic_ = get_parameter("output_image_with_detect_markers_topic").as_string();
        markers_topic_ = get_parameter("markers_topic").as_string();
        camera_info_path_ = get_parameter("camera_info_path").as_string();

        world_name_ = this->get_parameter("world_name").as_string();
        count_world_marker_ = this->get_parameter("count_world_marker").as_int();
        for (int i = 0; i < count_world_marker_; ++i) {
          int marker_id =
              this->get_parameter("id_world_marker_" + std::to_string(i)).as_int();
          ids_world_marker_.insert(marker_id);
          marker_id__coords_world_marker_[marker_id] = std::vector<cv::Point3f>(4);
          auto marker_coord =
              this->get_parameter("coord_world_marker_" + std::to_string(i))
                  .as_double_array();
          marker_id__coords_world_marker_[marker_id][0] =
              cv::Point3f(marker_coord[0], marker_coord[3], marker_coord[2]);
          marker_id__coords_world_marker_[marker_id][1] =
              cv::Point3f(marker_coord[3], marker_coord[4], marker_coord[5]);
          marker_id__coords_world_marker_[marker_id][2] =
              cv::Point3f(marker_coord[6], marker_coord[7], marker_coord[8]);
          marker_id__coords_world_marker_[marker_id][3] =
              cv::Point3f(marker_coord[9], marker_coord[10], marker_coord[11]);
        }

        count_object_ = this->get_parameter("count_object").as_int();
        for (int i = 0; i < count_object_; ++i) {
          int count_marker =
              this->get_parameter("count_object_" + std::to_string(i) + "_marker")
                  .as_int();
          object_i__count_marker_[i] = count_marker;
          object_i__marker_ids_[i] = std::unordered_set<int>();
          object_i__marker_ids__coords_[i] =
              std::unordered_map<int, std::vector<cv::Point3f>>();
          for (int j = 0; j < count_marker; ++j) {
            int marker_id = this->get_parameter("id_object_" + std::to_string(i) +
                                                "_marker_" + std::to_string(j))
                                .as_int();
            object_i__marker_ids_[i].insert(marker_id);
            object_i__marker_ids__coords_[i][marker_id] = std::vector<cv::Point3f>(4);
            auto marker_coord =
                this->get_parameter("coord_object_" + std::to_string(i) + "_marker_" +
                                    std::to_string(j))
                    .as_double_array();
            object_i__marker_ids__coords_[i][marker_id][0] =
                cv::Point3f(marker_coord[0], marker_coord[3], marker_coord[2]);
            object_i__marker_ids__coords_[i][marker_id][1] =
                cv::Point3f(marker_coord[3], marker_coord[4], marker_coord[5]);
            object_i__marker_ids__coords_[i][marker_id][2] =
                cv::Point3f(marker_coord[6], marker_coord[7], marker_coord[8]);
            object_i__marker_ids__coords_[i][marker_id][3] =
                cv::Point3f(marker_coord[9], marker_coord[10], marker_coord[11]);
          }
        }
    }

    void publishFrame(const cv::Mat& image)
    {
        const auto cur_now = now();
        cv::Mat imageCopy;
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        cv::aruco::detectMarkers(image, dictionary_, corners, ids);

        for (int i = 0; i < ids.size(); ++i) {
            RCLCPP_INFO(get_logger(), "detect marker %i", ids[i]);
        }

        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        }

        pose_evaluator::msg::CameraImageStamped image_msg;
        image_msg.header.stamp = cur_now;
        image_msg.header.frame_id = camera_params_.camera_id;
        image_msg.camera_info = cam_info_manager_->getCameraInfo();
        image_msg.camera_id = camera_params_.camera_id;
        auto image_cvb = cv_bridge::CvImage(image_msg.header, "bgr8", image).toImageMsg();
        image_msg.image = *image_cvb;
        image_with_detected_markers_publisher_->publish(image_msg);

        pose_evaluator::msg::CameraImageStamped image_with_detect_markers_msg;
        image_with_detect_markers_msg.header.stamp = cur_now;
        image_with_detect_markers_msg.header.frame_id = camera_params_.camera_id;
        image_with_detect_markers_msg.camera_info = cam_info_manager_->getCameraInfo();
        image_with_detect_markers_msg.camera_id = camera_params_.camera_id;
        auto image_with_detect_markers_cvb = cv_bridge::CvImage(image_with_detect_markers_msg.header, "bgr8", imageCopy).toImageMsg();
        image_with_detect_markers_msg.image = *image_with_detect_markers_cvb;
        image_with_detected_markers_publisher_->publish(image_with_detect_markers_msg);

        std::vector<pose_evaluator::msg::DetectedPoint> points;
        for (size_t i = 0; i < ids.size(); ++i) {
            int id = ids[i];
            if (ids_world_marker_.find(id) != ids_world_marker_.end()) {
                for (int j = 0; j < 4; ++j) {
                    pose_evaluator::msg::DetectedPoint point;
                    point.coordinate_frame = world_name_;
                    point.u = corners[i][j].x;
                    point.v = corners[i][j].y;
                    point.x = marker_id__coords_world_marker_[id][j].x;
                    point.y = marker_id__coords_world_marker_[id][j].y;
                    point.z = marker_id__coords_world_marker_[id][j].z;
                    points.push_back(point);
                }
            } else {
                for (int j = 0; j < count_object_; ++j) {
                    if (object_i__marker_ids_[j].find(id) != object_i__marker_ids_[j].end()) {
                        for (int k = 0; k < 4; ++k) {
                            pose_evaluator::msg::DetectedPoint point;
                            point.coordinate_frame = "object_" + std::to_string(j);
                            point.u = corners[i][k].x;
                            point.v = corners[i][k].y;
                            point.x = object_i__marker_ids__coords_[j][id][k].x;
                            point.y = object_i__marker_ids__coords_[j][id][k].y;
                            point.z = object_i__marker_ids__coords_[j][id][k].z;
                            points.push_back(point);
                        }
                    }
                }
            }
        }

        pose_evaluator::msg::DetectedPoints detected_points_msg;
        detected_points_msg.header.stamp = cur_now;
        detected_points_msg.header.frame_id = camera_params_.camera_id;
        // detected_points_msg.camera_info = msg->camera_info;
        detected_points_msg.camera_id = std::to_string(camera_params_.camera_id);
        detected_points_msg.points = points;
        markers_publisher_->publish(detected_points_msg);
    }
    
    CameraParams camera_params_;
    std::unique_ptr<CameraCapture> camera_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cam_info_manager_;

    std::string world_name_;
    int count_world_marker_;
    std::unordered_set<int> ids_world_marker_;
    std::unordered_map<int, std::vector<cv::Point3f>> marker_id__coords_world_marker_;

    int count_object_;
    std::unordered_map<int, int> object_i__count_marker_;
    std::unordered_map<int, std::unordered_set<int>> object_i__marker_ids_;
    std::unordered_map<int, std::unordered_map<int, std::vector<cv::Point3f>>>
      object_i__marker_ids__coords_;

    std::string markers_topic_;
    std::string output_image_topic_;
    std::string output_image_with_detect_markers_topic_;
    std::string camera_info_path_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    rclcpp::Publisher<pose_evaluator::msg::CameraImageStamped>::SharedPtr image_publisher_;
    rclcpp::Publisher<pose_evaluator::msg::CameraImageStamped>::SharedPtr image_with_detected_markers_publisher_;
    rclcpp::Publisher<pose_evaluator::msg::DetectedPoints>::SharedPtr markers_publisher_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
