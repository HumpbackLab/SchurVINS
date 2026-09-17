// SPDX-License-Identifier: GPL-3.0-or-later
// FAST-10 decision network adapted from fast_neon faster_corner_10_sse.cpp
// See the upstream fast_neon package notices for the original implementation.
#include <svo/direct/fast10_neon.h>
#if defined(SCHUR_OPT_FAST10_NEON) && SCHUR_OPT_FAST10_NEON
#if !defined(__aarch64__)
#error "SCHUR_OPT_FAST10_NEON requires AArch64"
#endif
#include <arm_neon.h>
#include <fast/corner_10.h>
#include <fast/faster_corner_utilities.h>
namespace svo {
void fast10DetectNEON(const uint8_t* img, int img_width, int img_height,
             int img_stride, short barrier, std::vector<fast::fast_xy>& corners) {
  // Saturating byte thresholds are equivalent only for nonnegative barriers.
  if (barrier < 0 || barrier > 255) {
    fast::fast_corner_detect_10(img, img_width, img_height, img_stride, barrier, corners);
    return;
  }
  const int w = img_stride, stride = 3*w;
  const uint8x8_t barriers = vdup_n_u8(barrier);
  for (int y=3; y<img_height-3; ++y) {
    int x=3;
    for (; x+8<=img_width-3; x+=8) {
      const uint8_t* p = img+y*img_stride+x;
      const uint8x8_t here = vld1_u8(p);
      const uint8x8_t lo = vqsub_u8(here, barriers);
      const uint8x8_t hi = vqadd_u8(here, barriers);
      uint8x16_t ans_b, ans_e;
      {
          uint8x8_t top = vld1_u8(p-stride);
          uint8x8_t bottom = vld1_u8(p+stride);

          ans_b = vcombine_u8(vclt_u8(top, lo), vcgt_u8(top, hi));
          ans_e = vcombine_u8(vclt_u8(bottom, lo), vcgt_u8(bottom, hi));
          if (!vmaxvq_u8(ans_b | ans_e))
              continue;
      }

      uint8x16_t ans_m, ans_p, possible;
      {
          uint8x8_t ul = vld1_u8(p-2-2*w);
          uint8x8_t lr = vld1_u8(p+2+2*w);
          ans_m = vcombine_u8(vclt_u8(ul, lo), vcgt_u8(ul, hi));
          ans_p = vcombine_u8(vclt_u8(lr, lo), vcgt_u8(lr, hi));
          possible = (ans_m & ans_b) | (ans_e & ans_p);
          if (!vmaxvq_u8(possible))
              continue;
      }

      uint8x16_t ans_o, ans_n;
      {
          uint8x8_t ll = vld1_u8(p-2+2*w);
          uint8x8_t ur = vld1_u8(p+2-2*w);
          ans_o = vcombine_u8(vclt_u8(ll, lo), vcgt_u8(ll, hi));
          ans_n = vcombine_u8(vclt_u8(ur, lo), vcgt_u8(ur, hi));
          possible &= ans_o | (ans_b & ans_n);
          possible &= ans_n | (ans_e & ans_o);
          if (!vmaxvq_u8(possible))
              continue;
      }

      uint8x16_t ans_h, ans_k;
      {
          uint8x8_t left = vld1_u8(p-3);
          uint8x8_t right = vld1_u8(p+3);
          ans_h = vcombine_u8(vclt_u8(left, lo), vcgt_u8(left, hi));
          ans_k = vcombine_u8(vclt_u8(right, lo), vcgt_u8(right, hi));
          possible &= ans_h | (ans_n & ans_k & ans_p);
          possible &= ans_k | (ans_m & ans_h & ans_o);
          if (!vmaxvq_u8(possible))
              continue;
      }

      uint8x16_t ans_a, ans_c;
      {
          uint8x8_t a = vld1_u8(p-1-stride);
          uint8x8_t c = vld1_u8(p+1-stride);
          ans_a = vcombine_u8(vclt_u8(a, lo), vcgt_u8(a, hi));
          ans_c = vcombine_u8(vclt_u8(c, lo), vcgt_u8(c, hi));
          possible &= ans_a | (ans_e & ans_p);
          possible &= ans_c | (ans_o & ans_e);
          if (!vmaxvq_u8(possible))
              continue;
      }

      uint8x16_t ans_d, ans_f;
      {
          uint8x8_t d = vld1_u8(p-1+stride);
          uint8x8_t f = vld1_u8(p+1+stride);
          ans_d = vcombine_u8(vclt_u8(d, lo), vcgt_u8(d, hi));
          ans_f = vcombine_u8(vclt_u8(f, lo), vcgt_u8(f, hi));
          const uint8x16_t ans_abc = ans_a & ans_b & ans_c;
          possible &= ans_d | (ans_abc & ans_n);
          possible &= ans_f | (ans_m & ans_abc);
          if (!vmaxvq_u8(possible))
              continue;
      }

      uint8x16_t ans_g, ans_i;
      {
          uint8x8_t g = vld1_u8(p-3-w);
          uint8x8_t ii = vld1_u8(p-3+w);
          ans_g = vcombine_u8(vclt_u8(g, lo), vcgt_u8(g, hi));
          ans_i = vcombine_u8(vclt_u8(ii, lo), vcgt_u8(ii, hi));
          possible &= ans_g | (ans_f & ans_p & ans_k);
          possible &= ans_i | (ans_c & ans_n & ans_k);
          if (!vmaxvq_u8(possible))
              continue;
      }

      uint8x16_t ans_j, ans_l;
      {
          uint8x8_t jj = vld1_u8(p+3-w);
          uint8x8_t l = vld1_u8(p+3+w);
          ans_j = vcombine_u8(vclt_u8(jj, lo), vcgt_u8(jj, hi));
          ans_l = vcombine_u8(vclt_u8(l, lo), vcgt_u8(l, hi));
          const uint8x16_t ans_ghi = ans_g & ans_h & ans_i;
          possible &= ans_j | (ans_d & ans_o & ans_ghi);
          possible &= ans_l | (ans_m & ans_a & ans_ghi);
          if (!vmaxvq_u8(possible))
              continue;
      }

      uint8_t hits[8];
      vst1_u8(hits, vget_low_u8(possible) | vget_high_u8(possible));
      for (int i=0; i<8; ++i)
        if (hits[i]) corners.emplace_back(x+i, y);
    }
    for (; x<img_width-3; ++x) {
      const uint8_t* p = img+y*img_stride+x;
      if (is_corner_10<fast::Less>(p, img_stride, barrier) ||
          is_corner_10<fast::Greater>(p, img_stride, barrier))
        corners.emplace_back(x,y);
    }
  }
}
void fast10ScoreNEON(const uint8_t* img, int stride,
            const std::vector<fast::fast_xy>& corners, int threshold,
            std::vector<int>& scores) {
  if (threshold < 0 || threshold > 255) {
    fast::fast_corner_score_10(img, stride, corners, threshold, scores);
    return;
  }
  const int offsets[16] = {3*stride,3*stride+1,2*stride+2,stride+3,
    3,3-stride,2-2*stride,1-3*stride,-3*stride,-1-3*stride,
    -2-2*stride,-3-stride,-3,stride-3,2*stride-2,3*stride-1};
  scores.resize(corners.size());
  for (std::size_t n=0; n<corners.size(); ++n) {
    const uint8_t* p = img+corners[n].y*stride+corners[n].x;
    int16_t diff[16];
    for (int i=0; i<16; ++i) diff[i] = int(p[offsets[i]]) - int(*p);
    const int16x8_t a = vld1q_s16(diff), b = vld1q_s16(diff+8);
    // Min/max over every circular 10-pixel arc, using 2/4/8/10 windows.
    const int16x8_t lo2a=vminq_s16(a,vextq_s16(a,b,1));
    const int16x8_t lo2b=vminq_s16(b,vextq_s16(b,a,1));
    const int16x8_t hi2a=vmaxq_s16(a,vextq_s16(a,b,1));
    const int16x8_t hi2b=vmaxq_s16(b,vextq_s16(b,a,1));
    const int16x8_t lo4a=vminq_s16(lo2a,vextq_s16(lo2a,lo2b,2));
    const int16x8_t lo4b=vminq_s16(lo2b,vextq_s16(lo2b,lo2a,2));
    const int16x8_t hi4a=vmaxq_s16(hi2a,vextq_s16(hi2a,hi2b,2));
    const int16x8_t hi4b=vmaxq_s16(hi2b,vextq_s16(hi2b,hi2a,2));
    const int16x8_t lo8a=vminq_s16(lo4a,vextq_s16(lo4a,lo4b,4));
    const int16x8_t lo8b=vminq_s16(lo4b,vextq_s16(lo4b,lo4a,4));
    const int16x8_t hi8a=vmaxq_s16(hi4a,vextq_s16(hi4a,hi4b,4));
    const int16x8_t hi8b=vmaxq_s16(hi4b,vextq_s16(hi4b,hi4a,4));
    const int16x8_t bright=vmaxq_s16(vminq_s16(lo8a,lo2b),vminq_s16(lo8b,lo2a));
    const int16x8_t dark=vnegq_s16(vminq_s16(vmaxq_s16(hi8a,hi2b),vmaxq_s16(hi8b,hi2a)));
    // Strict comparison: an arc's minimum contrast D passes at D-1.
    scores[n] = vmaxvq_s16(vmaxq_s16(bright,dark))-1;
  }
}
} // namespace svo
#endif
