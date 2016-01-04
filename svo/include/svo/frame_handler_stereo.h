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

#ifndef SVO_FRAME_HANDLER_STEREO_H_
#define SVO_FRAME_HANDLER_STEREO_H_

#include <set>
#include <vikit/abstract_camera.h>
#include <svo/frame_handler_base.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>

namespace svo {

/// Stereo Visual Odometry Pipeline
class FrameHandlerStereo : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FrameHandlerStereo(vk::AbstractCamera *cam_left, vk::AbstractCamera *cam_right, double baseline_px);
  virtual ~FrameHandlerStereo();

  /// Provide a stereo pair
  AddImageResult addImages(const cv::Mat& img_left, const cv::Mat& img_right, double timestamp);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(
      const int keyframe_id,
      const SE3& T_kf_f,
      const cv::Mat& img,
      const double timestamp);

protected:
  vk::AbstractCamera* cam_left_;                //!< Left camera model, can be ATAN, Pinhole or Ocam (see vikit).
  vk::AbstractCamera* cam_right_;               //!< Right camera model, can be ATAN, Pinhole or Ocam (see vikit).
  double baseline_px_;
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                     //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.

  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame(const SE3 &T_cur);

  /// Processes all frames after the first frame until a keyframe is selected.
  //virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const SE3& T_cur_ref,
      FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(double scene_depth_mean);

  void setCoreKfs(size_t n_closest);
};

} // namespace svo

#endif // SVO_FRAME_HANDLER_H_
