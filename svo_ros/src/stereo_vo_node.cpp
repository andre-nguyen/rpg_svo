// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2015 Kartik Mohta <kartikmohta@gmail.com>
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <ros/package.h>
#include <string>
#include <svo/frame_handler_stereo.h>
#include <svo/map.h>
#include <svo/config.h>
#include <svo/frame.h>
#include <svo_ros/visualizer.h>
#include <vikit/params_helper.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <vikit/abstract_camera.h>
#include <vikit/pinhole_camera.h>
#include <vikit/camera_loader.h>
#include <vikit/user_input_thread.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_geometry/stereo_camera_model.h>
#include "stereo_processor.h"

namespace svo {

/// SVO Interface
class StereoVoNode : public StereoProcessor
{
public:
  svo::FrameHandlerStereo* vo_;
  svo::Visualizer visualizer_;
  bool publish_markers_;                 //!< publish only the minimal amount of info (choice for embedded devices)
  bool publish_dense_input_;
  boost::shared_ptr<vk::UserInputThread> user_input_thread_;
  ros::Subscriber sub_remote_key_;
  std::string remote_input_;
  vk::AbstractCamera *l_cam_, *r_cam_;
  bool quit_;
  StereoVoNode(std::string transport);
  ~StereoVoNode();
  void imageCallback(const sensor_msgs::Image::ConstPtr &l_image_msg,
      const sensor_msgs::Image::ConstPtr &r_image_msg,
      const sensor_msgs::CameraInfo::ConstPtr &l_cinfo_msg,
      const sensor_msgs::CameraInfo::ConstPtr &r_cinfo_msg);
  void processUserActions();
  void remoteKeyCb(const std_msgs::StringConstPtr& key_input);

};

StereoVoNode::StereoVoNode(std::string transport) :
  vo_(NULL),
  StereoProcessor(transport),
  publish_markers_(vk::getParam<bool>("svo/publish_markers", true)),
  publish_dense_input_(vk::getParam<bool>("svo/publish_dense_input", false)),
  remote_input_(""),
  l_cam_(NULL),
  r_cam_(NULL),
  quit_(false)
{
  // Start user input thread in parallel thread that listens to console keys
  if(vk::getParam<bool>("svo/accept_console_user_input", false))
    user_input_thread_ = boost::make_shared<vk::UserInputThread>();

  // Get initial position and orientation
  visualizer_.T_world_from_vision_ = Sophus::SE3(
      vk::rpy2dcm(Vector3d(vk::getParam<double>("svo/init_rx", 0.0),
                           vk::getParam<double>("svo/init_ry", 0.0),
                           vk::getParam<double>("svo/init_rz", 0.0))),
      Eigen::Vector3d(vk::getParam<double>("svo/init_tx", 0.0),
                      vk::getParam<double>("svo/init_ty", 0.0),
                      vk::getParam<double>("svo/init_tz", 0.0)));
}

StereoVoNode::~StereoVoNode()
{
  delete vo_;
  delete l_cam_;
  delete r_cam_;
  if(user_input_thread_ != NULL)
    user_input_thread_->stop();
}

void StereoVoNode::imageCallback(
      const sensor_msgs::Image::ConstPtr &l_image_msg,
      const sensor_msgs::Image::ConstPtr &r_image_msg,
      const sensor_msgs::CameraInfo::ConstPtr &l_cinfo_msg,
      const sensor_msgs::CameraInfo::ConstPtr &r_cinfo_msg)
{
  cv::Mat l_img, r_img;
  try {
    l_img = cv_bridge::toCvShare(l_image_msg, "mono8")->image;
    r_img = cv_bridge::toCvShare(r_image_msg, "mono8")->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  //  configure camera calibration
  if (!l_cam_) {
    //  assume these parameters will not change throughout the odometry
    image_geometry::StereoCameraModel mdl;
    if (!mdl.fromCameraInfo(l_cinfo_msg, r_cinfo_msg)) {
      ROS_ERROR("Failed to initialize camera info");
      return;
    }
    std::cout << mdl.left().fx() << ", " << mdl.left().fy() << ", " <<
                 mdl.left().cx() << ", " << mdl.left().cy() << std::endl;
    l_cam_ = new vk::PinholeCamera(l_img.cols, l_img.rows,
                                   mdl.left().fx(), mdl.left().fy(),
                                   mdl.left().cx(), mdl.left().cy());
    r_cam_ = new vk::PinholeCamera(r_img.cols, r_img.rows,
                                   mdl.right().fx(), mdl.right().fy(),
                                   mdl.right().cx(), mdl.right().cy());
    ROS_INFO("Initialized camera model");

    //  now we can initialize the VO
    vo_ = new svo::FrameHandlerStereo(l_cam_, r_cam_, -mdl.right().Tx());
    vo_->start();
  }

  processUserActions();
  const FrameHandlerBase::AddImageResult res =
      vo_->addImages(l_img, r_img, l_image_msg->header.stamp.toSec());

  visualizer_.publishMinimal(l_img, vo_->lastFrame(), *vo_,
                             l_image_msg->header.stamp.toSec());

  if(publish_markers_ && vo_->stage() != FrameHandlerBase::STAGE_PAUSED)
    visualizer_.visualizeMarkers(vo_->lastFrame(), vo_->coreKeyframes(), vo_->map());

  if(publish_dense_input_)
    visualizer_.exportToDense(vo_->lastFrame());

  if(vo_->stage() == FrameHandlerStereo::STAGE_PAUSED)
    usleep(100000);
}

void StereoVoNode::processUserActions()
{
  char input = remote_input_.c_str()[0];
  remote_input_ = "";

  if(user_input_thread_ != NULL)
  {
    char console_input = user_input_thread_->getInput();
    if(console_input != 0)
      input = console_input;
  }

  switch(input)
  {
    case 'q':
      quit_ = true;
      printf("SVO user input: QUIT\n");
      break;
    case 'r':
      vo_->reset();
      printf("SVO user input: RESET\n");
      break;
    case 's':
      vo_->start();
      printf("SVO user input: START\n");
      break;
    default: ;
  }
}

void StereoVoNode::remoteKeyCb(const std_msgs::StringConstPtr& key_input)
{
  remote_input_ = key_input->data;
}

} // namespace svo


int main(int argc, char **argv)
{
  ros::init(argc, argv, "svo");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string transport;
  pnh.param("image_transport", transport, std::string("raw"));
  svo::StereoVoNode stereo_vo_node(transport);

  // subscribe to remote input
  stereo_vo_node.sub_remote_key_ = nh.subscribe("svo/remote_key", 5, &svo::StereoVoNode::remoteKeyCb, &stereo_vo_node);

  ros::spin();
  // start processing callbacks
  while(ros::ok() && !stereo_vo_node.quit_)
  {
    ros::spinOnce();
    // TODO check when last image was processed. when too long ago. publish warning that no msgs are received!
  }

  printf("SVO terminated.\n");
  return 0;
}
