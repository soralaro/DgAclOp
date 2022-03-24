
/*
 * Copyright (c) Deepglint Co., Ltd. 2022. All rights reserved.
 * by zhenxiongchen
 */

#ifndef _VEGA_WARP_AFFINE_KERNELS_H_
#define _VEGA_WARP_AFFINE_KERNELS_H_

#include "cpu_kernel.h"
#include <iostream>
#include "opencv2/opencv.hpp"
namespace aicpu {
class VegaWarpAffineCpuKernel : public CpuKernel {
public:
    ~VegaWarpAffineCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
