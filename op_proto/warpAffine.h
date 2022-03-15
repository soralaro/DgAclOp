/**
 * Copyright (C)  2022. Deepglint Co., Ltd. All rights reserved.
 * by zhenxiongchen@deepglint.com
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_WARP_AFFINE_H
#define GE_OP_WARP_AFFINE_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(WarpAffine)
    .INPUT(img_in, TensorType({DT_UINT8}))
    .INPUT(trans_M, TensorType({DT_FLOAT32}))
    .OUTPUT(img_out, TensorType({DT_UINT8}))
    .REQUIRED_ATTR(img_in_size, ListInt)
    .REQUIRED_ATTR(img_out_size, ListInt)
    .REQUIRED_ATTR(dst_size, ListInt)
    .OP_END_FACTORY_REG(WarpAffine)
}
#endif //GE_OP_WARP_AFFINE_H
