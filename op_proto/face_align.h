/**
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_FACE_ALIGN_H
#define GE_OP_FACE_ALIGN_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(FaceAlign)
    .INPUT(image, TensorType({DT_UINT8}))
    .INPUT(keypoints, TensorType({DT_FLOAT32}))
    .INPUT(face_num, TensorType({DT_INT32}))
    .OUTPUT(aligned_image, TensorType({DT_UINT8}))
    .REQUIRED_ATTR(face_size, ListInt)
    .REQUIRED_ATTR(default_keypoint, ListInt)
    .OP_END_FACTORY_REG(FaceAlign)
}
#endif //GE_OP_FACE_ALIGN_H
