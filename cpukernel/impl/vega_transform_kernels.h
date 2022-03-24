
/*
 * Copyright (c) Deepglint Co., Ltd. 2022. All rights reserved.
 * by zhenxiongchen
 */

#ifndef _VEGA_TRANSFORM_KERNELS_H_
#define _VEGA_TRANSFORM_KERNELS_H_

#include "cpu_kernel.h"
#include <iostream>
#include "opencv2/opencv.hpp"
namespace aicpu {
class VegaTransformCpuKernel : public CpuKernel {
public:
    ~VegaTransformCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
