#ifndef STEREO_PROCESSOR_H_
#define STEREO_PROCESSOR_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>

/**
 * This is an abstract base class for stereo image processing nodes.
 * It handles synchronization of input topics (approximate or exact)
 * and checks for sync errors.
 * To use this class, subclass it and implement the imageCallback() method.
 */
class StereoProcessor {
 private:
  // subscriber
  image_transport::SubscriberFilter l_image_sub_, r_image_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> l_cinfo_sub_,
      r_cinfo_sub_;
  typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
      sensor_msgs::CameraInfo> ExactPolicy;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
      sensor_msgs::CameraInfo> ApproximatePolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
  boost::shared_ptr<ExactSync> exact_sync_;
  boost::shared_ptr<ApproximateSync> approximate_sync_;
  int queue_size_;

  // for sync checking
  ros::WallTimer check_synced_timer_;
  int l_image_received_, r_image_received_, l_cinfo_received_,
      r_cinfo_received_, all_received_;

  // for sync checking
  static void increment(int* value) { ++(*value); }

  void dataCb(const sensor_msgs::ImageConstPtr& l_image_msg,
              const sensor_msgs::ImageConstPtr& r_image_msg,
              const sensor_msgs::CameraInfoConstPtr& l_info_msg,
              const sensor_msgs::CameraInfoConstPtr& r_info_msg) {
    // For sync error checking
    ++all_received_;

    // call implementation
    imageCallback(l_image_msg, r_image_msg, l_info_msg, r_info_msg);
  }

  void checkInputsSynchronized() {
    int threshold = 3 * all_received_;
    if (l_image_received_ >= threshold || r_image_received_ >= threshold ||
        l_cinfo_received_ >= threshold || r_cinfo_received_ >= threshold) {
      ROS_WARN(
          "[stereo_processor] Low number of synchronized "
          "left/right/left_info/right_info tuples received.\n"
          "Left images received:       %d (topic '%s')\n"
          "Right images received:      %d (topic '%s')\n"
          "Left camera info received:  %d (topic '%s')\n"
          "Right camera info received: %d (topic '%s')\n"
          "Synchronized tuples: %d\n"
          "Possible issues:\n"
          "\t* stereo_image_proc is not running.\n"
          "\t  Does `rosnode info %s` show any connections?\n"
          "\t* The cameras are not synchronized.\n"
          "\t  Try restarting the node with parameter _approximate_sync:=True\n"
          "\t* The network is too slow. One or more images are dropped from "
          "each tuple.\n"
          "\t  Try restarting the node, increasing parameter 'queue_size' "
          "(currently %d)",
          l_image_received_, l_image_sub_.getTopic().c_str(), r_image_received_,
          r_image_sub_.getTopic().c_str(), l_cinfo_received_,
          l_cinfo_sub_.getTopic().c_str(), r_cinfo_received_,
          r_cinfo_sub_.getTopic().c_str(), all_received_,
          ros::this_node::getName().c_str(), queue_size_);
    }
  }

 protected:
  /**
   * Constructor, subscribes to input topics using image transport and registers
   * callbacks.
   * \param transport The image transport to use
   */
  StereoProcessor(const std::string& transport)
      : l_image_received_(0),
        r_image_received_(0),
        l_cinfo_received_(0),
        r_cinfo_received_(0),
        all_received_(0) {
    // Read local parameters
    ros::NodeHandle local_nh("~");

    // Resolve topic names
    ros::NodeHandle nh;
    const std::string left = local_nh.resolveName("left");
    const std::string right = local_nh.resolveName("right");
    const std::string l_image_topic = local_nh.resolveName("left/image");
    const std::string r_image_topic = local_nh.resolveName("right/image");
    const std::string l_cinfo_topic = ros::names::append(left, "camera_info");
    const std::string r_cinfo_topic = ros::names::append(right, "camera_info");

    // Subscribe to four input topics.
    ROS_INFO("Subscribing to:\n\t* %s\n\t* %s\n\t* %s\n\t* %s",
             l_image_topic.c_str(), r_image_topic.c_str(),
             l_cinfo_topic.c_str(), r_cinfo_topic.c_str());

    image_transport::ImageTransport it(nh);
    l_image_sub_.subscribe(it, l_image_topic, 1, transport);
    r_image_sub_.subscribe(it, r_image_topic, 1, transport);
    l_cinfo_sub_.subscribe(nh, l_cinfo_topic, 1);
    r_cinfo_sub_.subscribe(nh, r_cinfo_topic, 1);

    // Complain every 15s if the topics appear unsynchronized
    l_image_sub_.registerCallback(
        boost::bind(StereoProcessor::increment, &l_image_received_));
    r_image_sub_.registerCallback(
        boost::bind(StereoProcessor::increment, &r_image_received_));
    l_cinfo_sub_.registerCallback(
        boost::bind(StereoProcessor::increment, &l_cinfo_received_));
    r_cinfo_sub_.registerCallback(
        boost::bind(StereoProcessor::increment, &r_cinfo_received_));
    check_synced_timer_ = nh.createWallTimer(
        ros::WallDuration(15.0),
        boost::bind(&StereoProcessor::checkInputsSynchronized, this));

    // Synchronize input topics. Optionally do approximate synchronization.
    local_nh.param("queue_size", queue_size_, 5);
    bool approx;
    local_nh.param("approximate_sync", approx, false);
    if (approx) {
      approximate_sync_.reset(
          new ApproximateSync(ApproximatePolicy(queue_size_), l_image_sub_,
                              r_image_sub_, l_cinfo_sub_, r_cinfo_sub_));
      approximate_sync_->registerCallback(
          boost::bind(&StereoProcessor::dataCb, this, _1, _2, _3, _4));
    } else {
      exact_sync_.reset(new ExactSync(ExactPolicy(queue_size_), l_image_sub_,
                                      r_image_sub_, l_cinfo_sub_,
                                      r_cinfo_sub_));
      exact_sync_->registerCallback(
          boost::bind(&StereoProcessor::dataCb, this, _1, _2, _3, _4));
    }
  }

  /**
   * Implement this method in sub-classes
   */
  virtual void imageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_cinfo_msg,
      const sensor_msgs::CameraInfoConstPtr& r_cinfo_msg) = 0;
};

#endif
