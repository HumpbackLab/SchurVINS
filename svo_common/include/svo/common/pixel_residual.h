// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <vikit/cameras/camera_geometry_base.h>
#include <cstdlib>
#include <cstring>
namespace svo {
inline bool pixelResidualEnabled() {
    const char* value = std::getenv("SCHUR_PIXEL_RESIDUAL");
    return value && std::strcmp(value, "1") == 0;
}
inline bool pixelResidual(const vk::cameras::CameraGeometryBase& camera,
                          const Eigen::Vector3d& bearing,
                          const Eigen::Vector3d& point,
                          Eigen::Vector2d& residual,
                          Eigen::Matrix<double, 2, 3>* jacobian = nullptr) {
    if (!bearing.allFinite() || !point.allFinite() || point.z() <= 1e-8)
        return false;
    Eigen::Vector2d measured, predicted;
    camera.project3(bearing, &measured);
    camera.project3(point, &predicted, jacobian);
    residual = measured - predicted;
    return residual.allFinite() && (!jacobian || jacobian->allFinite());
}
}  // namespace svo
