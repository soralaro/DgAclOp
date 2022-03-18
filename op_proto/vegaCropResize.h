/**
 * Copyright (C)  2022. Deepglint Co., Ltd. All rights reserved.
 * by zhenxiongchen@deepglint.com
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_VEGA_CROP_RESIZE_H
#define GE_OP_VEGA_CROP_RESIZE_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(VegaCropResize)
    .INPUT(data_in, TensorType({DT_UINT8}))
    .INPUT(param, TensorType({DT_UINT8}))
    .OUTPUT(data_out, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(VegaCropResize)
}
#endif //GE_OP_VEGA_CROP_RESIZE_H
