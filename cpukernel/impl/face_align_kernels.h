
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of FaceAlign
 */

#ifndef _FACE_ALIGN_KERNELS_H_
#define _FACE_ALIGN_KERNELS_H_

#include "cpu_kernel.h"
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
namespace aicpu {
class FaceAlignCpuKernel : public CpuKernel {
public:
    ~FaceAlignCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
