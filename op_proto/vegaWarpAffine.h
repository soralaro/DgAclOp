/**
 * Copyright (C)  2022. Deepglint Co., Ltd. All rights reserved.
 * by zhenxiongchen@deepglint.com
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_VEGA_WARP_AFFINE_H
#define GE_OP_VEGA_WARP_AFFINE_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(VegaWarpAffine)
    .INPUT(param, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(VegaWarpAffine)
}
#endif //GE_OP_VEGA_WARP_AFFINE_H
