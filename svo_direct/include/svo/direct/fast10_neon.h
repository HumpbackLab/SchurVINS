// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <fast/fast.h>
namespace svo {
// Same strict FAST-10 criterion, raster order and score convention as fast_neon.
void fast10DetectNEON(const uint8_t* image, int width, int height, int stride,
                      short threshold, std::vector<fast::fast_xy>& corners);
void fast10ScoreNEON(const uint8_t* image, int stride,
                     const std::vector<fast::fast_xy>& corners, int threshold,
                     std::vector<int>& scores);
}
