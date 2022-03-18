
/*
 * Copyright (c) Deepglint Co., Ltd. 2022. All rights reserved.
 * by zhenxiongchen@deepglint.com
 */
#include "vega_crop_resize_kernels.h"
#define CS_NONE 0
#define CS_RGB 1
#define CS_BGR 2
#define CS_UV  3
#define CS_VU  4
#define CS_YUV420P 6
#define PLANE_SHIFT 24
#define CS_SHIFT 16
#define UNIT_SHIFT 8
#define MAKE_TYPE(plane, cs, unit)  (((plane) << PLANE_SHIFT) | ((cs) << CS_SHIFT) | ((unit) << UNIT_SHIFT))

namespace  {
const char *VEGA_CROP_RESIZE = "VegaCropResize";
}
namespace aicpu  {
    enum vega_matrix_type {
        Undefined = 0,
        BGRPacked = MAKE_TYPE(1, CS_BGR, 3),
        RGBPacked = MAKE_TYPE(1, CS_RGB, 3),
        BGRPlanar = MAKE_TYPE(3, CS_BGR, 1),
        RGBPlanar = MAKE_TYPE(3, CS_RGB, 1),
        YUV420P = MAKE_TYPE(3, CS_YUV420P, 1),
        NV12 = MAKE_TYPE(2, CS_UV, 1),
        NV21 = MAKE_TYPE(2, CS_VU, 1),
        Gray = MAKE_TYPE(1, CS_NONE, 1)
    };
    typedef struct {
        unsigned long long img_in_addr;
        unsigned long long img_out_addr;
        unsigned  int img_in_w;
        unsigned  int img_in_h;
        unsigned  int img_in_roi_x;
        unsigned  int img_in_roi_y;
        unsigned  int img_in_roi_w;
        unsigned  int img_in_roi_h;
        unsigned  int img_out_w;
        unsigned  int img_out_h;
        unsigned  int img_out_roi_x;
        unsigned  int img_out_roi_y;
        unsigned  int img_out_roi_w;
        unsigned  int img_out_roi_h;
        unsigned  int img_in_type;
        unsigned  int img_out_type;
    }VegaCropResizeParam;
uint32_t VegaCropResizeCpuKernel::Compute(CpuKernelContext &ctx)
{

    Tensor *data_in_tensor = ctx.Input(0);
    if (data_in_tensor == nullptr) {
        return 1;
    }

    Tensor *param_tensor = ctx.Input(1);
    if (param_tensor == nullptr) {
        return 1;
    }

    Tensor *data_out_tensor = ctx.Output(0);
    if (data_out_tensor == nullptr) {
        return 1;
    }

    VegaCropResizeParam *param=(VegaCropResizeParam *)param_tensor->GetData();

    return 0;
}

REGISTER_CPU_KERNEL(VEGA_CROP_RESIZE, VegaCropResizeCpuKernel);
} // namespace aicpu