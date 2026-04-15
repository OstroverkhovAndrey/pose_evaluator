

#include <chrono>
#include <functional>
#include <string>

#include <unordered_map>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "rclcpp/rclcpp.hpp"
#include "pose_evaluator/msg/detected_point.hpp"
#include "pose_evaluator/msg/detected_points.hpp"
#include "pose_evaluator/filter_factory.hpp"
#include "pose_evaluator/white_noise_rigid_body_model.hpp"
#include "pose_evaluator/pinhole_point_measurement_model.hpp"
#include <sensor_msgs/msg/camera_info.hpp>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

using std::placeholders::_1;
using pose_evaluator::State;
using pose_evaluator::Cov12;
using pose_evaluator::IFilter;
using pose_evaluator::makeFilter;
using pose_evaluator::WhiteNoiseRigidBodyModel;
using pose_evaluator::CameraIntrinsics;
using pose_evaluator::PointObservation;
using pose_evaluator::PinholePointMeasurementModel;

class PoseEvaluatorNode : public rclcpp::Node {
public:
    PoseEvaluatorNode() : Node("pose_evaluator_node") {

        RCLCPP_INFO(this->get_logger(), "start cst");
        declare_parameters();
        get_parameters();

        auto process_model = std::make_shared<WhiteNoiseRigidBodyModel>(0.0, 0.0);
        world_filter_ = makeFilter("simple", process_model);
        for (int i = 0; i < count_object_; ++i) {
            object_name__filter_.insert({objects_name_[i], makeFilter("simple", process_model)});
        }

        // create subscription
        subscription_ = this->create_subscription<pose_evaluator::msg::DetectedPoints>(
                detected_points_topic_name_, 10, std::bind(&PoseEvaluatorNode::topic_callback, this, _1));

        // tf_broadcaster_ =
        //     std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // readCameraParameters("./src/aruco_localization/resources/cameraCalib.txt", camMatrix_, distCoeffs_);

        RCLCPP_INFO(this->get_logger(), "end cst");
    }

private:

    void declare_parameters() {
        this->declare_parameter("world_name", "world");
        this->declare_parameter("count_object", 1);
        for (int i = 0; i < this->get_parameter("count_object").as_int(); ++i) {
            this->declare_parameter("object_" + std::to_string(i) + "_name", "object" + std::to_string(i));
        }
        this->declare_parameter("detected_points_topic_name", "detected_points");
    }

    void get_parameters() {
        world_name_ = this->get_parameter("world_name").as_string();
        count_object_ = this->get_parameter("count_object").as_int();
        for (int i = 0; i < count_object_; ++i) {
            std::string object_i_name = this->get_parameter("object_" + std::to_string(i) + "_name").as_string();
            objects_name_.push_back(object_i_name);
        }
        detected_points_topic_name_ = this->get_parameter("detected_points_topic_name").as_string();
    }

    void create_observations(const pose_evaluator::msg::DetectedPoints & msg,
                             std::vector<PointObservation>& observations_w,
                             std::unordered_map<std::string, std::vector<PointObservation>>& object_name__observations_o) {

    //     std::vector<int> ids;
    //     std::vector<std::vector<cv::Point2f>> corners;

    //     for (int i = 0; i < (int)msg.markers.size(); ++i)
    //     {
    //         ids.push_back(msg.markers[i].marker_id);
    //         corners.push_back(std::vector<cv::Point2f>(4));
    //         corners[i][0].x = msg.markers[i].up_left.x;
    //         corners[i][0].y = msg.markers[i].up_left.y;
    //         corners[i][1].x = msg.markers[i].up_right.x;
    //         corners[i][1].y = msg.markers[i].up_right.y;
    //         corners[i][2].x = msg.markers[i].down_right.x;
    //         corners[i][2].y = msg.markers[i].down_right.y;
    //         corners[i][3].x = msg.markers[i].down_left.x;
    //         corners[i][3].y = msg.markers[i].down_left.y;
    //     }

    //     for (int i = 0; i < (int)ids.size(); ++i) {
    //       int id = ids[i];
    //       RCLCPP_INFO(this->get_logger(), "id %i", id);
    //       for (int j = 0; j < 4; ++j) {
    //         if (ids_world_marker_.find(id) != ids_world_marker_.end()) {
    //             PointObservation obs;
    //             obs.point_body = Eigen::Vector3d(
    //               coords_world_marker_[id][j].x,
    //               coords_world_marker_[id][j].y,
    //               coords_world_marker_[id][j].z
    //             );
    //             obs.pixel = Eigen::Vector2d(
    //                 corners[i][j].x,
    //                 corners[i][j].y
    //             );
    //             observations_w.push_back(obs);
    //         }
    //         if (ids_object_marker_.find(id) != ids_object_marker_.end()) {
    //             PointObservation obs;
    //             obs.point_body = Eigen::Vector3d(
    //               coords_object_marker_[id][j].x,
    //               coords_object_marker_[id][j].y,
    //               coords_object_marker_[id][j].z
    //             );
    //             obs.pixel = Eigen::Vector2d(
    //                 corners[i][j].x,
    //                 corners[i][j].y
    //             );
    //             observations_o.push_back(obs);
    //         }
    //       }
    //     }
    }

    void topic_callback(const pose_evaluator::msg::DetectedPoints & msg) {

        RCLCPP_INFO(this->get_logger(), "start callback");

        std::vector<PointObservation> observations_w;
        std::unordered_map<std::string, std::vector<PointObservation>> object_name__observations_o;
        create_observations(msg, observations_w, object_name__observations_o);

         
        world_filter_->predict(0.0);
        for (int i = 0; i < count_object_; ++i) {
            object_name__filter_[objects_name_[i]]->predict(0.0);
        }

        PinholePointMeasurementModel meas_model_w(K_, observations_w, sigma_px_);
        Eigen::VectorXd z_w = meas_model_w.measurementVector();
        world_filter_->update(z_w, meas_model_w);

        for (int i = 0; i < count_object_; ++i) {
            PinholePointMeasurementModel meas_model_o(K_, object_name__observations_o[objects_name_[i]], sigma_px_);
            Eigen::VectorXd z_o = meas_model_o.measurementVector();
            object_name__filter_[objects_name_[i]]->update(z_o, meas_model_o);
        }
    }

    std::string world_name_;
    int count_object_;
    std::vector<std::string> objects_name_;
    std::string detected_points_topic_name_;

    std::unique_ptr<IFilter> world_filter_;
    std::unordered_map<std::string, std::unique_ptr<IFilter>> object_name__filter_;


    rclcpp::Subscription<pose_evaluator::msg::DetectedPoints>::SharedPtr subscription_;
    int countMarkersWorldCoordinateSystem_;
    int countMarkersRobotCoordinateSystem_;
    std::unordered_map<int, std::vector<cv::Point3f>> markersWorldCoordinateSystem_;
    std::unordered_map<int, std::vector<cv::Point3f>> markersRobotCoordinateSystem_;
    cv::Mat camMatrix_, distCoeffs_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::string robot_name_;
    int i_ = 0;

    CameraIntrinsics K_;
    double sigma_px_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseEvaluatorNode>());
    rclcpp::shutdown();
    return 0;
}