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
 * \file images_ops_shape_fns.cpp
 * \brief
 */
#include "images_ops_shape_fns.h"
#include "op_log.h"
#include "error_util.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
graphStatus ColorspaceShapeFn(Operator& op, const std::string output_name) {
  Shape shape;
  graphStatus status = WithRankAtLeast(op.GetInputDesc(0), 1, shape, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    AscendString op_name;
    op.GetName(op_name);
    OP_LOGE(op_name.GetString(), "input[images] must 1-D or higher rank.");
    return GRAPH_PARAM_INVALID;
  }
  int64_t dim = op.GetInputDesc(0).GetShape().GetDims().back();
  if (dim != 3) {
    AscendString op_name;
    op.GetName(op_name);
    OP_LOGE(op_name.GetString(), "input[images] last dimension must be size 3.");
    return GRAPH_PARAM_INVALID;
  }
  TensorDesc desc = op.GetOutputDesc(output_name);
  desc.SetShape(Shape(shape));
  return op.UpdateOutputDesc(output_name, desc);
}

graphStatus ResizeShapeFn(Operator& op, const std::string input_name, const std::string size_input_name,
                          const std::string output_name) {
  Shape shape;
  graphStatus status = WithRank(op.GetInputDesc(0), 4, shape, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    AscendString op_name;
    op.GetName(op_name);
    OP_LOGE(op_name.GetString(), "input[images] must 4-D.");
    return GRAPH_PARAM_INVALID;
  }
  auto dims = op.GetInputDesc(0).GetShape().GetDims();
  auto channel_dim = dims[3];
  TensorDesc input_td = op.GetInputDesc(0);
  if (input_td.GetFormat() == FORMAT_NCHW) {
    channel_dim = dims[1];
  }
  return SetOutputToSizedImage(op, dims[0], size_input_name, channel_dim, output_name);
}

graphStatus SetOutputToSizedImage(Operator& op, const int64_t batch_dim, const std::string size_input_name,
                                  const int64_t channel_dim, const std::string output_name) {
  Shape size_shape;
  graphStatus status = WithRank(op.GetInputDesc(size_input_name), 1, size_shape, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    AscendString op_name;
    op.GetName(op_name);
    OP_LOGE(op_name.GetString(), "input size must be 1-D.");
    return GRAPH_PARAM_INVALID;
  }
  auto size_dims = op.GetInputDesc(size_input_name).GetShape().GetDims();
  if (size_dims[0] != 2) {
    AscendString op_name;
    op.GetName(op_name);
    OP_LOGE(op_name.GetString(), "input size must be 1-D tensor of 2 elements.");
    return GRAPH_PARAM_INVALID;
  }

  std::vector<std::string> input_infer_depends = {size_input_name};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor size_tensor;
  TensorDesc td = op.GetOutputDesc(output_name);
  status = op.GetInputConstData(size_input_name, size_tensor);
  if (status != GRAPH_SUCCESS) {
    td.SetDataType(DT_FLOAT);
    std::vector<int64_t> out_shape;
    TensorDesc input_td = op.GetInputDesc(0);
    if (input_td.GetFormat() == FORMAT_NCHW) {
      out_shape.push_back(batch_dim);
      out_shape.push_back(channel_dim);
      out_shape.push_back(-1);
      out_shape.push_back(-1);
    } else if (input_td.GetFormat() == FORMAT_NHWC) {
      out_shape.push_back(batch_dim);
      out_shape.push_back(-1);
      out_shape.push_back(-1);
      out_shape.push_back(channel_dim);
    } else {
      std::string error_msg = "Not supported this format";
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), error_msg);
    }
    td.SetShape(Shape(out_shape));
    op.UpdateOutputDesc(output_name, td);
    return GRAPH_SUCCESS;
  }

  const int32_t* size_data = reinterpret_cast<const int32_t*>(size_tensor.GetData());

  int64_t size_width = static_cast<int64_t>(size_data[1]);
  int64_t size_height = static_cast<int64_t>(size_data[0]);
  std::vector<int64_t> output_shape;

  TensorDesc input_td = op.GetInputDesc(0);
  if (input_td.GetFormat() == FORMAT_NCHW) {
    output_shape.push_back(batch_dim);
    output_shape.push_back(channel_dim);
    output_shape.push_back(size_height);
    output_shape.push_back(size_width);
  } else if (input_td.GetFormat() == FORMAT_NHWC) {
    output_shape.push_back(batch_dim);
    output_shape.push_back(size_height);
    output_shape.push_back(size_width);
    output_shape.push_back(channel_dim);
  } else {
    OP_LOGE(op.GetName().c_str(), "Not supported this format");
  }
  td.SetShape(Shape(output_shape));
  return op.UpdateOutputDesc(output_name, td);
}

graphStatus EncodeImageShapeFn(Operator& op) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 3, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AscendString op_name;
    op.GetName(op_name);
    OP_LOGE(op_name.GetString(), "input rank must be 3 .");
    return GRAPH_FAILED;
  }

  Shape output_shape;
  (void)Scalar(output_shape);
  TensorDesc output_tensor = op.GetOutputDesc("contents");
  output_tensor.SetDataType(DT_STRING);
  output_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("contents", output_tensor);
}

graphStatus DecodeImageShapeFn(Operator& op) {
  int channels;
  if (op.GetAttr("channels", channels) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("Get attr[chanels] failed"));
    return GRAPH_FAILED;
  }
  if (channels != 0 && channels != 1 && channels != 3 && channels != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("attr[Channels] must be 0,1,3,or 4"));
    return GRAPH_FAILED;
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("Get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims;
  if (channels == 0) {
     dims = {ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, ge::UNKNOWN_DIM};
  } else {
    dims = {ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, channels};
  }
  
  Shape output_shape(dims);
  TensorDesc output_tensor = op.GetOutputDesc(0);
  output_tensor.SetDataType(dtype);
  output_tensor.SetShape(output_shape);
  if (op.UpdateOutputDesc("image", output_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("Update OutputDesc[image] failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool DimsAllEqualOrUnknown(std::initializer_list<int64_t>&& inputs, int64_t unknown_dim_val) {
  auto it = inputs.begin();
  for (; it != inputs.end() && (*it == unknown_dim_val); ++it) {
  }

  if (it == inputs.end()) {
    return true;
  }

  for (auto default_dim_val = *(it++); it != inputs.end(); ++it) {
    if (*it != default_dim_val && *it != unknown_dim_val) {
      return false;
    }
  }

  return true;
}

}  // namespace ge