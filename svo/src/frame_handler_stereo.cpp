// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

// Modification Note: 
// This file may have been modified by the authors of SchurVINS.
// (All authors of SchurVINS are with PICO department of ByteDance Corporation)

#include <svo/frame_handler_stereo.h>
#include <svo/map.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/pose_optimizer.h>
#include <svo/img_align/sparse_img_align.h>
#include <svo/direct/depth_filter.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>
#include <vikit/performance_monitor.h>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <unordered_map>

namespace svo {
namespace {
size_t rejectStereoEpipolarOutliers(const FrameBundlePtr& bundle,
                                   double max_error_px) {
  const FramePtr& left = bundle->at(0);
  const FramePtr& right = bundle->at(1);
  const Transformation T_right_left =
      right->T_cam_world() * left->T_world_cam();
  const Eigen::Matrix3d R = T_right_left.getRotationMatrix();
  const Eigen::Vector3d t = T_right_left.getPosition();
  const double focal = left->cam()->getIntrinsicParameters()[0];
  std::unordered_map<int, size_t> left_index;
  for (size_t i = 0; i < left->numFeatures(); ++i)
    if (left->landmark_vec_[i])
      left_index.emplace(left->landmark_vec_[i]->id(), i);

  size_t rejected = 0;
  for (size_t j = 0; j < right->numFeatures(); ++j)
  {
    const PointPtr& point = right->landmark_vec_[j];
    if (!point)
      continue;
    const auto it = left_index.find(point->id());
    if (it == left_index.end())
      continue;
    const size_t i = it->second;
    if (!left->landmark_vec_[i])
      continue;
    const Eigen::Vector3d x0 = left->f_vec_.col(i) / left->f_vec_(2, i);
    const Eigen::Vector3d x1 = right->f_vec_.col(j) / right->f_vec_(2, j);
    const Eigen::Vector3d line1 = t.cross(R * x0);
    const Eigen::Vector3d line0 = R.transpose() * x1.cross(t);
    const double denominator = std::sqrt(
        line0.head<2>().squaredNorm() + line1.head<2>().squaredNorm());
    const double error_px = denominator > 1e-12
        ? focal * std::abs(x1.dot(line1)) / denominator
        : std::numeric_limits<double>::infinity();
    if (!std::isfinite(error_px) || error_px > max_error_px)
    {
      left->type_vec_[i] = FeatureType::kOutlier;
      right->type_vec_[j] = FeatureType::kOutlier;
      left->landmark_vec_[i] = nullptr;
      right->landmark_vec_[j] = nullptr;
      ++rejected;
    }
  }
  return rejected;
}
}

FrameHandlerStereo::FrameHandlerStereo(
    const BaseOptions& base_options,
    const DepthFilterOptions& depth_filter_options,
    const DetectorOptions& feature_detector_options,
    const InitializationOptions& init_options,
    const StereoTriangulationOptions& stereo_options,
    const ReprojectorOptions& reprojector_options,
    const FeatureTrackerOptions& tracker_options,
    const CameraBundle::Ptr& stereo_camera)
  : FrameHandlerBase(
      base_options, reprojector_options, depth_filter_options,
      feature_detector_options, init_options, tracker_options, stereo_camera)
{
  // init initializer
  stereo_triangulation_.reset(
        new StereoTriangulation(
          stereo_options,
          feature_detection_utils::makeDetector(
            feature_detector_options, cams_->getCameraShared(0))));
}

UpdateResult FrameHandlerStereo::processFrameBundle()
{
  UpdateResult res = UpdateResult::kFailure;
  if(stage_ == Stage::kTracking)
    res = processFrame();
  else if(stage_ == Stage::kInitializing)
    res = processFirstFrame();
  return res;
}

void FrameHandlerStereo::addImages(
    const cv::Mat& img_left,
    const cv::Mat& img_right,
    const uint64_t timestamp)
{
  // TODO: deprecated
  addImageBundle({img_left, img_right}, timestamp);
}

UpdateResult FrameHandlerStereo::processFirstFrame()
{
  schurvinsForward();
  if(initializer_->addFrameBundle(new_frames_) == InitResult::kFailure)
  {
    SVO_ERROR_STREAM("Initialization failed. Not enough triangulated points.");
    return UpdateResult::kDefault;
  }

  new_frames_->at(0)->setKeyframe();
  map_->addKeyframe(new_frames_->at(0),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  new_frames_->at(1)->setKeyframe();
  map_->addKeyframe(new_frames_->at(1),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);

  frame_utils::getSceneDepth(new_frames_->at(0), depth_median_, depth_min_, depth_max_);
  depth_filter_->addKeyframe(new_frames_->at(0), depth_median_, 0.5*depth_min_, depth_median_*1.5);
  schurvinsBackward();

  LOG(INFO) << std::fixed << "InitState: " << new_frames_->getMinTimestampSeconds()
            << ", quat: " << new_frames_->get_T_W_B().getEigenQuaternion().w() << ", "
            << new_frames_->get_T_W_B().getEigenQuaternion().x() << ", "
            << new_frames_->get_T_W_B().getEigenQuaternion().y() << ", "
            << new_frames_->get_T_W_B().getEigenQuaternion().z() << ", "
            << "pos: " << new_frames_->get_T_W_B().getPosition()[0] << ", "
            << new_frames_->get_T_W_B().getPosition()[1] << ", " << new_frames_->get_T_W_B().getPosition()[2];  

  SVO_INFO_STREAM("Init: Selected first frame.");
  stage_ = Stage::kTracking;
  tracking_quality_ = TrackingQuality::kGood;
  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerStereo::processFrame()
{
  // ---------------------------------------------------------------------------
  // tracking

  // STEP 1: Sparse Image Align
  size_t n_tracked_features = 0;
  sparseImageAlignment();

  // STEP 2: Map Reprojection & Feature Align
  n_tracked_features = projectMapInFrame();
  if (const char* value = std::getenv("SCHUR_STEREO_EPIPOLAR_PX"))
  {
    const double threshold_px = std::atof(value);
    if (std::isfinite(threshold_px) && threshold_px > 0.0)
    {
      const size_t rejected = rejectStereoEpipolarOutliers(
          new_frames_, threshold_px);
      n_tracked_features -= std::min(n_tracked_features, rejected * 2);
    }
  }
  if(n_tracked_features < options_.quality_min_fts)
  {
    tracking_quality_ = TrackingQuality::kInsufficient;
    return makeKeyframe(); // force stereo triangulation to recover
  }

  // STEP 3: Pose & Structure Optimization
  if(bundle_adjustment_type_!=BundleAdjustmentType::kCeres)
  {
    n_tracked_features = optimizePose();
    if(n_tracked_features < options_.quality_min_fts)
    {
      tracking_quality_ = TrackingQuality::kInsufficient;
      return makeKeyframe(); // force stereo triangulation to recover
    }
    optimizeStructure(new_frames_, options_.structure_optimization_max_pts, 5);
  }
  // return if tracking bad
  setTrackingQuality(n_tracked_features);
  if(tracking_quality_ == TrackingQuality::kInsufficient)
  {
    return makeKeyframe(); // force stereo triangulation to recover
  }

  // ---------------------------------------------------------------------------
  // select keyframe
  frame_utils::getSceneDepth(new_frames_->at(0), depth_median_, depth_min_, depth_max_);
  if(!need_new_kf_(new_frames_->at(0)->T_f_w_))
  {
    for(size_t i=0; i<new_frames_->size(); ++i)
      depth_filter_->updateSeeds(overlap_kfs_.at(i), new_frames_->at(i));
    return UpdateResult::kDefault;
  }
  SVO_DEBUG_STREAM("New keyframe selected.");
  return makeKeyframe();
}

UpdateResult FrameHandlerStereo::makeKeyframe()
{
  static size_t kf_counter = 0;
  const size_t kf_id = kf_counter++ % cams_->numCameras();
  const size_t other_id = kf_counter % cams_->numCameras();
  CHECK(kf_id != other_id);

  // ---------------------------------------------------------------------------
  // add extra features when num tracked is critically low!
  if(new_frames_->numLandmarks() < options_.kfselect_numkfs_lower_thresh)
  {
    setDetectorOccupiedCells(0, stereo_triangulation_->feature_detector_);
    new_frames_->at(other_id)->setKeyframe();
    map_->addKeyframe(new_frames_->at(other_id),
                      bundle_adjustment_type_==BundleAdjustmentType::kCeres);
    upgradeSeedsToFeatures(new_frames_->at(other_id));
    stereo_triangulation_->compute(new_frames_->at(0), new_frames_->at(1));
  }

  // ---------------------------------------------------------------------------
  // new keyframe selected
  new_frames_->at(kf_id)->setKeyframe();
  map_->addKeyframe(new_frames_->at(kf_id),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  upgradeSeedsToFeatures(new_frames_->at(kf_id));

  // init new depth-filters, set feature-detection grid-cells occupied that
  // already have a feature
  {
    DepthFilter::ulock_t lock(depth_filter_->feature_detector_mut_);
    setDetectorOccupiedCells(kf_id, depth_filter_->feature_detector_);
  } // release lock
  depth_filter_->addKeyframe(
        new_frames_->at(kf_id), depth_median_, 0.5*depth_min_, depth_median_*1.5);
  depth_filter_->updateSeeds(overlap_kfs_.at(0), new_frames_->at(0));
  depth_filter_->updateSeeds(overlap_kfs_.at(1), new_frames_->at(1));

  // TEST
  // {
  if(options_.update_seeds_with_old_keyframes)
  {
    depth_filter_->updateSeeds({ new_frames_->at(0) }, last_frames_->at(0));
    depth_filter_->updateSeeds({ new_frames_->at(0) }, last_frames_->at(1));
    depth_filter_->updateSeeds({ new_frames_->at(1) }, last_frames_->at(0));
    depth_filter_->updateSeeds({ new_frames_->at(1) }, last_frames_->at(1));
    for(const FramePtr& old_keyframe : overlap_kfs_.at(0))
    {
      depth_filter_->updateSeeds({ new_frames_->at(0) }, old_keyframe);
      depth_filter_->updateSeeds({ new_frames_->at(1) }, old_keyframe);
    }
  }
  // }

  // if limited number of keyframes, remove the one furthest apart
  while(map_->size() > options_.max_n_kfs && options_.max_n_kfs > 2)
  {
    if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
    {
      // deal differently with map for ceres backend
      map_->removeOldestKeyframe();
    }
    else
    {
      FramePtr furthest_frame =
          map_->getFurthestKeyframe(new_frames_->at(kf_id)->pos());
      map_->removeKeyframe(furthest_frame->id());
    }
  }
  return UpdateResult::kKeyframe;
}

void FrameHandlerStereo::resetAll()
{
  backend_scale_initialized_ = true;
  resetVisionFrontendCommon();
}

} // namespace svo
