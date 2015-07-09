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
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_stereo.h>
#include <svo/initialization.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>

#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif

namespace svo
{
FrameHandlerStereo::FrameHandlerStereo(vk::AbstractCamera *cam_left,
                                       vk::AbstractCamera *cam_right,
                                       double baseline_px)
    : FrameHandlerBase(),
      cam_left_(cam_left),
      cam_right_(cam_right),
      baseline_px_(baseline_px),
      reprojector_(cam_left_, map_),
      depth_filter_(NULL)
{
  initialize();
}

void FrameHandlerStereo::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_left_->width(), cam_left_->height(), Config::gridSize(),
          Config::nPyrLevels()));
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  depth_filter_->startThread();
}

FrameHandlerStereo::~FrameHandlerStereo()
{
  delete depth_filter_;
}

FrameHandlerBase::AddImageResult FrameHandlerStereo::addImages(
    const cv::Mat &img_left, const cv::Mat &img_right, double timestamp)
{
  if(!startFrameProcessingCommon(timestamp))
    return RESULT_NORMAL;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  SVO_START_TIMER("pyramid_creation");
  new_frame_.reset(new Frame(cam_left_, img_left.clone(), cam_right_,
                             img_right.clone(), baseline_px_, timestamp));
  SVO_STOP_TIMER("pyramid_creation");

  // process frame
  UpdateResult res = RESULT_FAILURE;
  AddImageResult ai_res = FrameHandlerBase::RESULT_NORMAL;
  if(stage_ == STAGE_DEFAULT_FRAME)
  {
    res = processFrame();
  }
  else if(stage_ == STAGE_FIRST_FRAME)
  {
    res = processFirstFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()));
    if(res == RESULT_IS_KEYFRAME)
    {
      // insert the first keyframe
      ai_res = FrameHandlerBase::RESULT_ADDED_FIRST;
    }
  }
  else if(stage_ == STAGE_RELOCALIZING)
  {
    //res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
    //                      map_.getClosestKeyframe(last_frame_));
    SVO_WARN_STREAM("Relocalizing");
    res = processFirstFrame(last_frame_->T_f_w_);
  }

  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
  return ai_res;
}

FrameHandlerStereo::UpdateResult FrameHandlerStereo::processFirstFrame(const SE3 &T_cur_ref)
{
  new_frame_->T_f_w_ = T_cur_ref;
  // vector<cv::Point2f> px_left;
  // initialization::detectFeatures(new_left_frame_, px_left_, f_left_);
  Features new_features;
  feature_detection::FastDetector detector(
      new_frame_->img().cols, new_frame_->img().rows, Config::gridSize(), 1);
  detector.detect(new_frame_.get(), new_frame_->img_pyr_,
                  Config::triangMinCornerScore(), new_features);

  vector<double> depths;
  vector<double> depth_errors;
  SVO_INFO_STREAM("FrameHandlerStereo processFirstFrame() getStereoDepths");
  DepthFilter::getStereoDepths(new_frame_, new_features, depths, depth_errors);

  vector<double> depth_vec;
  double depth_min = std::numeric_limits<double>::max();
  for(const auto &depth : depths)
  {
    if(depth > 0)
    {
      depth_min = std::min(depth, depth_min);
      depth_vec.push_back(depth);
    }
  }
  SVO_INFO_STREAM("Init: KLT tracked " << depth_vec.size() << " features");

  // if(depth_vec.size() < Config::initMinTracked())
  if(depth_vec.size() < 20)
  {
    SVO_WARN_STREAM("Cannot set scene depth. Not enough triangulated points!");
    return RESULT_FAILURE;
  }
  double depth_mean = vk::getMedian(depth_vec);

  // For each inlier create 3D point and add feature in both frames
  const SE3 T_world_cur = new_frame_->T_f_w_.inverse();
  int idx = 0;
  for(auto &ftr : new_features)
  {
    const auto &depth = depths[idx];
    if(depth > 0)
    {
      const Vector2d px_cur{ftr->px(0), ftr->px(1)};
      const Vector3d f_cur{ftr->f};
      const Vector3d xyz_in_cur{f_cur * depth / f_cur(2)};
      const Vector3d pos = T_world_cur * xyz_in_cur;
      Point *new_point = new Point(pos);
      Feature *ftr_cur(
          new Feature(new_frame_.get(), new_point, px_cur, f_cur, ftr->level));
      new_frame_->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur);
    }
    delete ftr;
    ftr = nullptr;
    idx++;
  }
  new_frame_->setKeyframe();
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5 * depth_min);

  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  SVO_INFO_STREAM("Init: Triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerStereo::processFrame()
{
  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(), 30,
                           SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = " << repr_n_mps
                                                << "\t \t nMatches = "
                                                << repr_n_new_references);
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }

  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false, new_frame_,
      sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "
                   << sfba_error_init << "px\t thresh = " << sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = " << sfba_error_final
                                                 << "px\t nObsFin. = "
                                                 << sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(),
                    Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(Features::iterator it = new_frame_->fts_.begin();
      it != new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

// optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_, loba_n_erredges_init,
                loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init,
             loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"
                     << loba_n_erredges_init << ", " << loba_n_erredges_fin
                     << "} \t "
                        "Error {" << loba_err_init << ", " << loba_err_fin
                     << "}");
  }
#endif

  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5 * depth_min);

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the
    // mapper thread, maybe we
    // can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);

  return RESULT_IS_KEYFRAME;
}

FrameHandlerStereo::UpdateResult FrameHandlerStereo::relocalizeFrame(
    const SE3 &T_cur_ref, FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(), 30,
                           SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  if(img_align_n_tracked > 30)
  {
    SE3 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;
    FrameHandlerStereo::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerStereo::relocalizeFrameAtPose(const int keyframe_id,
                                               const SE3 &T_f_kf,
                                               const cv::Mat &img,
                                               const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_left_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE)
  {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerStereo::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerStereo::setFirstFrame(const FramePtr &first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerStereo::needNewKf(double scene_depth_mean)
{
  for(auto it = overlap_kfs_.begin(), ite = overlap_kfs_.end(); it != ite; ++it)
  {
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x()) / scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y()) / scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.z()) / scene_depth_mean < Config::kfSelectMinDist())
      return false;
  }
  return true;
}

void FrameHandlerStereo::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size() - 1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin() + n,
                    overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                        boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(),
                [&](pair<FramePtr, size_t> &i)
                {
                  core_kfs_.insert(i.first);
                });
}

} // namespace svo
