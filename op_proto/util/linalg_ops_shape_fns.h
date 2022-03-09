/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file linalg_ops_shape_fns.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_LINALG_OPS_SHAPE_FNS_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_LINALG_OPS_SHAPE_FNS_H_

#include "graph/tensor.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"

namespace ge {

/**
 * Generate a square matrix's Shape
 * @param tensor Input tensor
 * @param out Output Shape
 * @return status whether this operation success
 */
graphStatus MakeBatchSquareMatrix(const TensorDesc& tensor, Shape& out, const char* op_name);
/**
 * Generate a square matrix's Shape
 * @param tensor Input ge tensor desc ptr
 * @param out Output GeShape
 * @return status whether this operation success
 */
graphStatus MakeBatchSquareMatrix(const GeTensorDescPtr& tensor_desc,
                                  GeShape& out, const char* op_name);
/**
 * Solving linear equations from matrices common shape func
 * @param tensor1 first input tensor
 * @param tensor2 second input tensor
 * @param square whether matrix is square
 * @param out Output Shape
 * @return status whether this operation success
 */
graphStatus MatrixSolve(const TensorDesc& tensor1, const TensorDesc& tensor2, bool square, Shape& out,
                        const char* op_name);

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_LINALG_OPS_SHAPE_FNS_H_
