
// слушает топик публикующий картинки, находит на картинке 
// aruco маркеры и публикует их координаты

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include "pose_evaluator/msg/detected_point.hpp"
#include "pose_evaluator/msg/detected_points.hpp"
#include "pose_evaluator/msg/camera_image_stamped.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.hpp>

using std::placeholders::_1;

class DetectAruco : public rclcpp::Node {
public:
    DetectAruco() : Node("aruco_detector_node") {

        declare_parameters();
        get_parameters();

        image_subscription_ = this->create_subscription<pose_evaluator::msg::CameraImageStamped>(
                input_topic_, 10, std::bind(
                &DetectAruco::topic_callback, this, _1));
        image_publisher_ = this->create_publisher<pose_evaluator::msg::CameraImageStamped>(
                output_topic_, 10);
        markers_publisher_ = this->create_publisher<pose_evaluator::msg::DetectedPoints>(
                markers_topic_, 10);
        
        dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    }
private:

    void declare_parameters() {
        declare_parameter<std::string>("input_topic", "/image_with_undetect_markers");
        declare_parameter<std::string>("output_topic", "/image_with_detect_markers");
        declare_parameter<std::string>("markers_topic", "/coordinates_markers_in_image");

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
        input_topic_ = get_parameter("input_topic").as_string();
        output_topic_ = get_parameter("output_topic").as_string();
        markers_topic_ = get_parameter("markers_topic").as_string();

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

    void topic_callback(const pose_evaluator::msg::CameraImageStamped::SharedPtr msg) /*const*/ {

        cv::Mat image = cv_bridge::toCvCopy(msg->image, "bgr8")->image;
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

        auto cv_ptr_marked = cv_bridge::CvImage(msg->image.header, "bgr8", imageCopy).toImageMsg();
        pose_evaluator::msg::CameraImageStamped marked_msg;
        marked_msg.header = msg->header;
        marked_msg.camera_info = msg->camera_info;
        marked_msg.image = *cv_ptr_marked;
        image_publisher_->publish(marked_msg);

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
        detected_points_msg.header = msg->header;
        detected_points_msg.camera_info = msg->camera_info;
        detected_points_msg.points = points;
        markers_publisher_->publish(detected_points_msg);
    }

    std::string world_name_;
    int count_world_marker_;
    std::unordered_set<int> ids_world_marker_;
    std::unordered_map<int, std::vector<cv::Point3f>> marker_id__coords_world_marker_;

    int count_object_;
    std::unordered_map<int, int> object_i__count_marker_;
    std::unordered_map<int, std::unordered_set<int>> object_i__marker_ids_;
    std::unordered_map<int, std::unordered_map<int, std::vector<cv::Point3f>>>
      object_i__marker_ids__coords_;

    std::string input_topic_, output_topic_, markers_topic_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    rclcpp::Subscription<pose_evaluator::msg::CameraImageStamped>::SharedPtr image_subscription_;
    rclcpp::Publisher<pose_evaluator::msg::CameraImageStamped>::SharedPtr image_publisher_;
    rclcpp::Publisher<pose_evaluator::msg::DetectedPoints>::SharedPtr markers_publisher_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectAruco>());
    rclcpp::shutdown();
    return 0;
}
