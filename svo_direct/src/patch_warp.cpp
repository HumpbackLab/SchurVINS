// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/patch_warp.h>
#if defined(SCHUR_OPT_WARP_AFFINE_NEON) && SCHUR_OPT_WARP_AFFINE_NEON
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#endif
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/common/feature_wrapper.h>
#include <svo/common/camera.h>
#include <glog/logging.h>

namespace svo {
namespace warp {

// TODO(cfo) take inverse depth!
void getWarpMatrixAffine(
    const CameraPtr& cam_ref,
    const CameraPtr& cam_cur,
    const Eigen::Ref<Keypoint>& px_ref,
    const Eigen::Ref<BearingVector>& f_ref,
    const double depth_ref,
    const Transformation& T_cur_ref,
    const int level_ref,
    AffineTransformation2* A_cur_ref)
{
  CHECK_NOTNULL(A_cur_ref);

  // Compute affine warp matrix A_ref_cur
  const int kHalfPatchSize = 5;
  const Position xyz_ref = f_ref * depth_ref;
  Position xyz_du_ref, xyz_dv_ref;
  // NOTE: project3 has no guarantee that the returned vector is unit length
  // - for pinhole: z component is 1 (unit plane)
  // - for omnicam: norm is 1 (unit sphere)
  cam_ref->backProject3(px_ref + Eigen::Vector2d(kHalfPatchSize,0) * (1<<level_ref), &xyz_du_ref);
  cam_ref->backProject3(px_ref + Eigen::Vector2d(0,kHalfPatchSize) * (1<<level_ref), &xyz_dv_ref);
  if(cam_ref->getType() == Camera::Type::kPinhole)
  {
    xyz_du_ref *= xyz_ref[2];
    xyz_dv_ref *= xyz_ref[2];
  }
  else
  {
    xyz_du_ref.normalize();
    xyz_dv_ref.normalize();
    xyz_du_ref *= depth_ref;
    xyz_dv_ref *= depth_ref;
  }

  Keypoint px_cur, px_du_cur, px_dv_cur;
  cam_cur->project3(T_cur_ref * xyz_ref, &px_cur);
  cam_cur->project3(T_cur_ref * xyz_du_ref, &px_du_cur);
  cam_cur->project3(T_cur_ref * xyz_dv_ref, &px_dv_cur);
  A_cur_ref->col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref->col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void getWarpMatrixAffineHomography(
    const CameraPtr& cam_ref,
    const CameraPtr& cam_cur,
    const Keypoint& px_ref,
    const BearingVector& f_ref,
    const BearingVector& normal_ref,
    const double depth_ref,
    const Transformation T_cur_ref,
    const int level_ref,
    AffineTransformation2& A_cur_ref)
{
  // normal = frame_ref_->T_f_w_.rotation_matrix()*point->normal_;
  // create homography matrix
  const Position t = T_cur_ref.inverse().getPosition();
  const BearingVector xyz_ref(f_ref*depth_ref);
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.getRotationMatrix()*
      (normal_ref.dot(xyz_ref)*Eigen::Matrix3d::Identity() - t*normal_ref.transpose());

  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 5;
  const BearingVector f_cur(H_cur_ref * f_ref);
  BearingVector f_du_ref, f_dv_ref;
  cam_ref->backProject3(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1<<level_ref), &f_du_ref);
  cam_ref->backProject3(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1<<level_ref), &f_dv_ref);
  const BearingVector f_du_cur = H_cur_ref * f_du_ref;
  const BearingVector f_dv_cur = H_cur_ref * f_dv_ref;
  Keypoint px_cur, px_du_cur, px_dv_cur;
  cam_cur->project3(f_cur, &px_cur);
  cam_cur->project3(f_du_cur, &px_du_cur);
  cam_cur->project3(f_dv_cur, &px_dv_cur);
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

int getBestSearchLevel(
    const AffineTransformation2& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

bool warpAffine(
    const AffineTransformation2& A_cur_ref,
    const cv::Mat& img_ref,
    const Eigen::Ref<Keypoint>& px_ref,
    const int level_ref,
    const int search_level,
    const int halfpatch_size,
    uint8_t* patch)
{
  Eigen::Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>()*(1<<search_level);
  if(std::isnan(A_ref_cur(0,0)))
  {
    LOG(WARNING) << "Affine warp is NaN, probably camera has no translation";
    return false;
  }

  // Perform the warp on a larger patch.
  uint8_t* patch_ptr = patch;
  const Eigen::Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  const int stride = img_ref.step.p[0];
#if defined(SCHUR_OPT_WARP_AFFINE_NEON) && SCHUR_OPT_WARP_AFFINE_NEON
  // The matcher uses a 10x10 patch. Other sizes and overlapping buffers keep
  // the scalar contract. In particular, SIMD must not read future source
  // pixels before an overlapping output write changes them.
  const uintptr_t dst_begin = reinterpret_cast<uintptr_t>(patch);
  const uintptr_t src_begin = reinterpret_cast<uintptr_t>(img_ref.data);
  const uintptr_t src_end = src_begin + (img_ref.rows > 0 ? (img_ref.rows-1)*img_ref.step[0]+img_ref.cols : 0);
  if(halfpatch_size == 5 && img_ref.rows > 0 && img_ref.cols > 0 &&
     (dst_begin+100 <= src_begin || dst_begin >= src_end) &&
     A_ref_cur.allFinite() && px_ref_pyr.allFinite())
  {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float xs[4] = {-5.0f, -4.0f, -3.0f, -2.0f};
    for(int y=-5; y<5; ++y)
    {
      // Do not fold px_ref_pyr into the row term: that changes rounding.
      const float32x4_t row_x = vdupq_n_f32(A_ref_cur(0,1)*y);
      const float32x4_t row_y = vdupq_n_f32(A_ref_cur(1,1)*y);
      for(int x=-5; x<5; x+=4)
      {
#if defined(SCHUR_OPT_WARP_TAIL2) && SCHUR_OPT_WARP_TAIL2
        if(x==3)
        {
          const float32x2_t xv=vadd_f32(vget_low_f32(vld1q_f32(xs)),vdup_n_f32(8.0f));
          const float32x2_t px=vadd_f32(vfma_n_f32(vget_low_f32(row_x),xv,A_ref_cur(0,0)),vdup_n_f32(px_ref_pyr[0]));
          const float32x2_t py=vadd_f32(vfma_n_f32(vget_low_f32(row_y),xv,A_ref_cur(1,0)),vdup_n_f32(px_ref_pyr[1]));
          const int32x2_t ix=vcvtm_s32_f32(px), iy=vcvtm_s32_f32(py);
          const uint32x2_t valid=vand_u32(vand_u32(vcge_s32(ix,vdup_n_s32(0)),vcge_s32(iy,vdup_n_s32(0))),
              vand_u32(vclt_s32(ix,vdup_n_s32(img_ref.cols-1)),vclt_s32(iy,vdup_n_s32(img_ref.rows-1))));
          if(vget_lane_u32(vpmin_u32(valid,valid),0)==0) goto scalar_warp;
          const uint8_t* p0=img_ref.data+vget_lane_s32(iy,0)*stride+vget_lane_s32(ix,0);
          const uint8_t* p1=img_ref.data+vget_lane_s32(iy,1)*stride+vget_lane_s32(ix,1);
          const auto pair=[](const uint8_t* p)->uint16_t {return p[0]|(uint16_t(p[1])<<8);};
          uint16x4_t top=vdup_n_u16(pair(p0)),bottom=vdup_n_u16(pair(p0+stride));
          top=vset_lane_u16(pair(p1),top,1);bottom=vset_lane_u16(pair(p1+stride),bottom,1);
          const uint32x2_t t=vget_low_u32(vmovl_u16(top)), b=vget_low_u32(vmovl_u16(bottom)), mask=vdup_n_u32(255);
          const float32x2_t tl=vcvt_f32_u32(vand_u32(t,mask)), tr=vcvt_f32_u32(vshr_n_u32(t,8));
          const float32x2_t bl=vcvt_f32_u32(vand_u32(b,mask)), br=vcvt_f32_u32(vshr_n_u32(b,8));
          const float32x2_t sx=vsub_f32(px,vcvt_f32_s32(ix)), sy=vsub_f32(py,vcvt_f32_s32(iy)), one2=vdup_n_f32(1.0f);
          const float32x2_t w00=vmul_f32(vsub_f32(one2,sx),vsub_f32(one2,sy));
          const float32x2_t w01=vmul_f32(vsub_f32(one2,sx),sy), w10=vmul_f32(sx,vsub_f32(one2,sy));
          const float32x2_t w11=vsub_f32(vsub_f32(vsub_f32(one2,w00),w01),w10);
          float32x2_t value=vfma_f32(vmul_f32(w01,bl),w00,tl);
          value=vfma_f32(value,w10,tr);value=vfma_f32(value,w11,br);
          const uint32x2_t out=vcvt_u32_f32(value);
          patch_ptr[0]=static_cast<uint8_t>(vget_lane_u32(out,0));
          patch_ptr[1]=static_cast<uint8_t>(vget_lane_u32(out,1));
          patch_ptr+=2;
          continue;
        }
#endif
        const int count = x == 3 ? 2 : 4;
        const float32x4_t xv = vaddq_f32(vld1q_f32(xs), vdupq_n_f32(x+5));
        const float32x4_t px = vaddq_f32(vfmaq_n_f32(row_x, xv, A_ref_cur(0,0)), vdupq_n_f32(px_ref_pyr[0]));
        const float32x4_t py = vaddq_f32(vfmaq_n_f32(row_y, xv, A_ref_cur(1,0)), vdupq_n_f32(px_ref_pyr[1]));
        const int32x4_t ix = vcvtmq_s32_f32(px), iy = vcvtmq_s32_f32(py);
        uint32x4_t valid = vandq_u32(vandq_u32(vcgeq_s32(ix,vdupq_n_s32(0)), vcgeq_s32(iy,vdupq_n_s32(0))),
            vandq_u32(vcltq_s32(ix,vdupq_n_s32(img_ref.cols-1)), vcltq_s32(iy,vdupq_n_s32(img_ref.rows-1))));
        if(count == 2) { valid=vsetq_lane_u32(~0u,valid,2); valid=vsetq_lane_u32(~0u,valid,3); }
        if(vminvq_u32(valid) == 0) goto scalar_warp;
        const uint8_t* p0 = img_ref.data + vgetq_lane_s32(iy,0)*stride + vgetq_lane_s32(ix,0);
        const uint8_t* p1 = img_ref.data + vgetq_lane_s32(iy,1)*stride + vgetq_lane_s32(ix,1);
        const uint8_t* p2 = count == 4 ? img_ref.data + vgetq_lane_s32(iy,2)*stride + vgetq_lane_s32(ix,2) : p0;
        const uint8_t* p3 = count == 4 ? img_ref.data + vgetq_lane_s32(iy,3)*stride + vgetq_lane_s32(ix,3) : p1;
        // Gather adjacent byte pairs, then unpack/interpolate in registers.
        // Both bytes were validated above. No speculative wide image loads.
        const auto pair = [](const uint8_t* p) -> uint16_t { return p[0] | (uint16_t(p[1]) << 8); };
        uint16x4_t top=vdup_n_u16(pair(p0)), bottom=vdup_n_u16(pair(p0+stride));
        top=vset_lane_u16(pair(p1),top,1); bottom=vset_lane_u16(pair(p1+stride),bottom,1);
        top=vset_lane_u16(pair(p2),top,2); bottom=vset_lane_u16(pair(p2+stride),bottom,2);
        top=vset_lane_u16(pair(p3),top,3); bottom=vset_lane_u16(pair(p3+stride),bottom,3);
        const uint32x4_t t=vmovl_u16(top), b=vmovl_u16(bottom), mask=vdupq_n_u32(255);
        const float32x4_t tl=vcvtq_f32_u32(vandq_u32(t,mask)), tr=vcvtq_f32_u32(vshrq_n_u32(t,8));
        const float32x4_t bl=vcvtq_f32_u32(vandq_u32(b,mask)), br=vcvtq_f32_u32(vshrq_n_u32(b,8));
        const float32x4_t sx = vsubq_f32(px,vcvtq_f32_s32(ix));
        const float32x4_t sy = vsubq_f32(py,vcvtq_f32_s32(iy));
        const float32x4_t w00 = vmulq_f32(vsubq_f32(one,sx),vsubq_f32(one,sy));
        const float32x4_t w01 = vmulq_f32(vsubq_f32(one,sx),sy);
        const float32x4_t w10 = vmulq_f32(sx,vsubq_f32(one,sy));
        const float32x4_t w11 = vsubq_f32(vsubq_f32(vsubq_f32(one,w00),w01),w10);
        float32x4_t value = vfmaq_f32(vmulq_f32(w01,bl),w00,tl);
        value = vfmaq_f32(value,w10,tr);
        value = vfmaq_f32(value,w11,br);
        const uint16x4_t out16=vmovn_u32(vcvtq_u32_f32(value));
        const uint8x8_t out8=vmovn_u16(vcombine_u16(out16,out16));
        // Byte lanes retain their order on either endianness.
        uint8_t out[8]; vst1_u8(out,out8);
        if(count==4) std::memcpy(patch_ptr,out,4); else std::memcpy(patch_ptr,out,2);
        patch_ptr += count;
      }
    }
    return true;
  }
scalar_warp:
  patch_ptr = patch;
#endif
  for (int y=-halfpatch_size; y<halfpatch_size; ++y)
  {
    for (int x=-halfpatch_size; x<halfpatch_size; ++x, ++patch_ptr)
    {
      const Eigen::Vector2f px_patch(x, y);
      const Eigen::Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
      const int xi = std::floor(px[0]);
      const int yi = std::floor(px[1]);
      if (xi<0 || yi<0 || xi+1>=img_ref.cols || yi+1>=img_ref.rows)
        return false;
      else
      {
        const float subpix_x = px[0]-xi;
        const float subpix_y = px[1]-yi;
        const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
        const float w01 = (1.0f-subpix_x)*subpix_y;
        const float w10 = subpix_x*(1.0f-subpix_y);
        const float w11 = 1.0f - w00 - w01 - w10;
        const uint8_t* const ptr = img_ref.data + yi*stride + xi;
        *patch_ptr = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride+1]);
      }
    }
  }
  return true;
}

bool warpPixelwise(
    const Frame& cur_frame,
    const Frame& ref_frame,
    const FeatureWrapper& ref_ftr,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    uint8_t* patch)
{
  double depth_ref = (ref_frame.pos() - ref_ftr.landmark->pos()).norm();
  double depth_cur = (cur_frame.pos() - ref_ftr.landmark->pos()).norm();

  // backproject to 3D points in reference frame
  Eigen::Vector3d xyz_ref;
  ref_frame.cam()->backProject3(ref_ftr.px, &xyz_ref);
  xyz_ref = xyz_ref.normalized() * depth_ref;

  // project to current frame and convert to search level
  Eigen::Vector3d xyz_cur = cur_frame.T_cam_world()*(ref_frame.T_cam_world().inverse())*xyz_ref;
  Eigen::Vector2d px_cur;
  cur_frame.cam()->project3(xyz_cur, &px_cur);
  Eigen::Vector2d px_cur_search = px_cur / (1<<level_cur);

  // for each pixel in the patch(on search level):
  // - convert to image level
  // - backproject to 3D points
  // - project to ref frame and find pixel value in ref level
  uint8_t* patch_ptr = patch;
  const cv::Mat& img_ref = ref_frame.img_pyr_[level_ref];
  const int stride = img_ref.step.p[0];

  for(int y=-halfpatch_size; y<halfpatch_size; ++y)
  {
    for(int x=-halfpatch_size; x<halfpatch_size; ++x, ++patch_ptr)
    {
      const Eigen::Vector2d ele_patch(x, y);
      Eigen::Vector2d ele_search = ele_patch + px_cur_search;
      Eigen::Vector3d ele_xyz_cur;
      cur_frame.cam()->backProject3(ele_search*(1<<level_cur), &ele_xyz_cur);
      ele_xyz_cur = ele_xyz_cur.normalized() * depth_cur;
      Eigen::Vector3d ele_xyz_ref = ref_frame.T_cam_world() * (cur_frame.T_cam_world().inverse()) * ele_xyz_cur;
      Eigen::Vector2d ele_ref;
      ref_frame.cam()->project3(ele_xyz_ref, &ele_ref);
      ele_ref = ele_ref / (1<<level_ref);

      const int xi = std::floor(ele_ref[0]);
      const int yi = std::floor(ele_ref[1]);
      if (xi<0 || yi<0 || xi+1>=img_ref.cols || yi+1>=img_ref.rows)
      {
        VLOG(200) << "ref image: col-" << img_ref.cols
                  << ", row-" << img_ref.rows;
        VLOG(200) << "xi: " << xi << ", " << "yi: " << yi;
        return false;
      }
      else
      {
        const float subpix_x = ele_ref[0]-xi;
        const float subpix_y = ele_ref[1]-yi;
        const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
        const float w01 = (1.0f-subpix_x)*subpix_y;
        const float w10 = subpix_x*(1.0f-subpix_y);
        const float w11 = 1.0f - w00 - w01 - w10;
        const uint8_t* const ptr = img_ref.data + yi*stride + xi;
        *patch_ptr = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride+1]);
      }
    }
  }

  return true;
}

void createPatchNoWarp(
    const cv::Mat& img,
    const Eigen::Vector2i& px,
    const int halfpatch_size,
    uint8_t* patch)
{
  CHECK_NOTNULL(patch);
  CHECK(px(0) >= halfpatch_size
        && px(1) >= halfpatch_size
        && px(0) < img.cols - halfpatch_size
        && px(1) < img.rows - halfpatch_size);

  const int patch_size = 2 * halfpatch_size;
  uint8_t* patch_ptr = patch;
  const int step = img.step.p[0];
  for (int y = 0; y < patch_size; ++y)
  {
    uint8_t* img_ptr =
        (uint8_t*) img.data
        + (px[1] - halfpatch_size + y) * step
        + (px[0] - halfpatch_size);
    for (int x = 0; x < patch_size; ++x, ++patch_ptr, ++img_ptr)
    {
      *patch_ptr = *img_ptr;
    }
  }
}

void createPatchNoWarpInterpolated(
    const cv::Mat& img,
    const Eigen::Ref<Keypoint>& px,
    const int halfpatch_size,
    uint8_t* patch)
{
  CHECK_NOTNULL(patch);

  // TODO(cfo): This could be easily implemented using SIMD instructions.

  const int step = img.step.p[0];
  const float u = px(0);
  const float v = px(1);
  const int u_r = std::floor(u);
  const int v_r = std::floor(v);
  CHECK(u_r >= halfpatch_size
        && v_r >= halfpatch_size
        && u_r < img.cols - halfpatch_size
        && v_r < img.rows - halfpatch_size);

  // compute interpolation weights
  const float subpix_x = u - u_r;
  const float subpix_y = v - v_r;
  const float wTL = (1.0-subpix_x) * (1.0-subpix_y);
  const float wTR = subpix_x * (1.0-subpix_y);
  const float wBL = (1.0-subpix_x)*subpix_y;
  const float wBR = subpix_x * subpix_y;

  const int patch_size = 2 * halfpatch_size;
  uint8_t* patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    uint8_t* img_ptr =
        (uint8_t*) img.data
        + (v_r - halfpatch_size + y) * step
        + (u_r - halfpatch_size);
    for (int x = 0; x < patch_size; ++x, ++patch_ptr, ++img_ptr)
    {
      *patch_ptr =
          wTL*img_ptr[0] + wTR*img_ptr[1] + wBL*img_ptr[step] + wBR*img_ptr[step+1];;
    }
  }
}

} // namespace warp
} // namespace svo
