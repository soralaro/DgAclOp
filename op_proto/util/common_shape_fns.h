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
 * \file common_shape_fns.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_COMMON_SHAPE_FNS_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_COMMON_SHAPE_FNS_H_

#include <string>
#include <vector>
#include "graph/tensor.h"
#include "graph/operator.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "graph/resource_context.h"
#include "error_code.h"

namespace ge {

struct ShapeAndRange {
  Shape shape_;
  std::vector<std::pair<int64_t, int64_t>> shape_range_;
};

struct AicpuResourceContext : ResourceContext {
  std::vector<ShapeAndRange> shape_and_range_;
};

/**
 * Check whether Shape's rank is at least rank
 * @param tensor Input tensor
 * @param rank expect val of Shape
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRankAtLeast(const TensorDesc& tensor, int64_t rank, Shape& out, const char* op_name);

/**
 * Check whether Shape's rank is at least rank
 * @param tensor Input tensor
 * @param rank expect val of Shape
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRankAtLeast(const GeTensorDescPtr& tensorDesc, int64_t rank, GeShape& out_shape,
                            const char* op_name);

/**
 * Check whether Shape's rank is equal to rank
 * @param shape Input tensor shape
 * @param rank expect shape rank
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRankShape(GeShape& shape, int64_t rank, const char* op_name);

/**
 * Check whether Shape's rank is equal to rank
 * @param tensor Input tensor
 * @param rank expect val of Shape
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRank(const TensorDesc& tensor, int64_t rank, Shape& out, const char* op_name);

/**
 * Check whether Shape's rank is equal to rank
 * @param tensor Input tensor
 * @param rank expect val of Shape
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRank(const GeTensorDescPtr& tensorDesc, int64_t rank,
                     GeShape& out_shape, const char* op_name);

/**
 * Check whether Shape's rank is equal to rank
 * @param tensor Input tensor
 * @param rank expect val of Shape
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRank(const GeTensorDescPtr& tensorDesc, int64_t rank,
                     Shape& out_shape, const char* op_name);

/**
 * Check whether dim is equal to value
 * @param dim Input dim
 * @param value expect val of dim
 * @param out Output dim
 * @return status whether Dim is equal to value
 */
graphStatus WithValue(int64_t dim, int64_t value, int64_t& out, const char* op_name);

/**
 * Merge two shapes
 * @param s0 first shape val
 * @param prefix second shape val
 * @param s_out merged shape val
 * @param prefix_out prefix out shape val
 * @return status whether this operation success
 */
graphStatus MergePrefix(const Shape& s, const Shape& prefix, Shape &s_out,
                        Shape &prefix_out, const char* op_name);


/**
 * Merge two dims of Shape
 * @param dim0 first dim val
 * @param dim1 second dim val
 * @param out merged dim val
 * @return status whether this operation success
 */
graphStatus Merge(int64_t dim1, int64_t dim2, int64_t& out);

/**
 * Merge two shapes
 * @param s0 first shape val
 * @param s1 second shape val
 * @param out merged shape val
 * @return status whether this operation success
 */
graphStatus Merge(const Shape& s0, const Shape& s1, Shape& out, const char* op_name);

/**
 * Merge two shapes
 * @param s0 first Geshape val
 * @param s1 second Geshape val
 * @param out merged Geshape val
 * @return status whether this operation success
 */
graphStatus Merge(const GeShape& s0, const GeShape& s1, GeShape& out, const char* op_name);

/**
 * Merge two shapes
 * @param shared_shape first Geshape val
 * @param value_shape second Geshape val
 * @param out merged shape val
 * @param shape_changed status whether shape has changed
 */
void MergeShape(const Shape &shared_shape, const Shape &value_shape,
                std::vector<int64_t> &out, bool &shape_changed);

/**
 * Merge two shape ranges
 * @param shared_shape_range first shape range val
 * @param value_shape_range second shape range val
 * @param out merged shape range val
 * @param shape_changed status whether shape range has changed
 */
void MergeRange(const std::vector<std::pair<int64_t, int64_t>> &shared_shape_range,
                const std::vector<std::pair<int64_t, int64_t>> &value_shape_range,
                std::vector<std::pair<int64_t, int64_t>> &out, bool &shape_changed);

/**
 * Merge two shapes and ranges
 * @param shared_shape_and_range first shape and range val
 * @param value_shape_and_range second shape and range val
 * @param out merged shape and range val
 * @param shape_changed status whether shape and range has changed
 * @return status whether this operation success
 */
graphStatus MergeShapeAndRange(const ShapeAndRange &shared_shape_and_range,
                               const ShapeAndRange &value_shape_and_range,
                               ShapeAndRange &out,
                               bool &shape_changed,
                               const char* op_name);

/**
 * Replace one dim in a given shape
 * @param s original shape
 * @param dim_index_in dim index
 * @param new_dim new dim value
 * @param out new shape
 * @return status whether this operation success
 */
graphStatus ReplaceDim(const Shape& s, int64_t dim_index_in, int64_t new_dim, Shape& out, const char* op_name);

/**
 * Replace one dim in a given shape
 * @param s original shape
 * @param dim_index_in dim index
 * @param new_dim new dim value
 * @param out new shape
 * @return status whether this operation success
 */
graphStatus ReplaceDim(const GeShape& s, int64_t dim_index_in, int64_t new_dim, GeShape& out, const char* op_name);

/**
 * Check if it satisfies 0 <= index < limit
 * @param index first input
 * @param limit second input
 * @return status whether this operation success
 */
template <typename Ta, typename Tb>
bool FastBoundsCheck(const Ta index, const Tb limit);

/**
 * Add two dims
 * @param dim0 first dim val
 * @param dim1 second dim val
 * @param out sum dim val
 * @return status whether this operation success
 */
graphStatus Add(int64_t dim1, int64_t dim2, int64_t& out);

/**
 * Subtract two dims
 * @param dim0 first dim val
 * @param dim1 second dim val
 * @param out Subtract dim val
 * @return status whether this operation success
 */
graphStatus Subtract(int64_t dim1, int64_t dim2, int64_t& out, const char* op_name);

/**
 * Get SubShape according to start end index and step size stride
 * @param s input Shape
 * @param start sub start index
 * @param end sub end index
 * @param stride sub step size
 * @param out sub shape output
 * @return status whether this operation success
 */
graphStatus SubShape(const Shape& s, int64_t start, int64_t end, int64_t stride, Shape& out, const char* op_name);

/**
 * Get SubShape according to start end index and step size stride
 * @param s input Shape
 * @param start sub start index
 * @param end sub end index
 * @param stride sub step size
 * @param out sub shape output
 * @return status whether this operation success
 */
graphStatus SubShape(const GeShape& s, size_t start, size_t end, size_t stride, GeShape& out);

/**
 * Get SubShape according to start end index and step size stride
 * @param s input Shape
 * @param start sub start index
 * @param end sub end index
 * @param stride sub step size
 * @param out sub shape output
 * @return status whether this operation success
 */
graphStatus SubShape(const GeShape& s, int64_t start, int64_t end, int64_t stride, GeShape& out, const char* op_name);

/**
 * Concatenate two shape
 * @param s1 first shape
 * @param s2 second shape
 * @param out concatenated shape
 * @return status whether this operation success
 */
graphStatus Concatenate(const Shape& s1, const Shape& s2, Shape& out);

/**
 * Concatenate two shape
 * @param s1 first shape
 * @param s2 second shape
 * @param out concatenated shape
 * @return status whether this operation success
 */
graphStatus Concatenate(const GeShape& s1, const GeShape& s2, GeShape& out);

/**
 * Gen matrix shape according d1 and d2
 * @param dim1 first dim val
 * @param dim2 first dim val
 * @param out matrix shape
 * @return status whether this operation success
 */
graphStatus Matrix(int64_t dim1, int64_t dim2, Shape& out);

/**
 * Gen vector shape according d
 * @param dim dim val
 * @param out vector shape
 * @return status whether this operation success
 */
graphStatus Vector(int64_t dim, Shape& out);

/**
 * Make shape from shape tensor
 * @param tensor shape tensor
 * @param out shape
 * @return status whether this operation success
 */
graphStatus MakeShapeFromShapeTensor(const Tensor& tensor, Shape& out, const char* op_name);

/**
 * Make shape from shape tensor
 * @param op Operator
 * @param dst_name const string &
 * @param out GeShape
 * @param op_name const char *
 * @return status whether this operation success
 */
graphStatus MakeShapeFromShapeTensor(Operator& op, const string& dst_name, GeShape& out, const char* op_name);

/**
 * Make dim from scalar tensor
 * @param tensor shape tensor
 * @param out shape
 * @return status whether this operation success
 */
graphStatus MakeDimForScalarInput(const Tensor& tensor, int64_t& out, const char* op_name);

/**
 * Check whether Shape's rank is at most rank
 * @param tensor input tensor
 * @param rank expect val of Shape
 * @param out output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRankAtMost(const TensorDesc& tensor, int64_t rank, Shape& out, const char* op_name);

/**
 * Check whether Shape's rank is at most rank
 * @param tensor input tensor
 * @param rank expect val of Shape
 * @param out output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus WithRankAtMost(const GeTensorDescPtr& tensorDesc, int64_t rank,
                           GeShape& out_shape, const char* op_name);

/**
 * make a empty dim shape
 * @param out output Shape
 * @return status whether Shape's condition Satisfied
 */
graphStatus Scalar(Shape& out);

/**
 * set input_name shape to output_name shape
 * @param op Operator which need to infershape
 * @param input_name input name of Operator
 * @param output_name ouput name of Operator
 * @return status whether infershape success
 */
graphStatus UnchangedShape(Operator& op, const string input_name, const string output_name);

/**
 * Devide dim
 * @param dividend
 * @param divisor
 * @param evenlyDivisible if to be divisible
 * @param out dims
 * @return status whether this operation success
 */
graphStatus Divide(const int64_t dividend, const int64_t divisor, const bool evenlyDivisible, int64_t& out,
                   const char* op_name);

/**
 * check shape fully defined or not
 * @param shape Shape is checked
 * @return whether shape is fully defined
 */
bool ShapeFullDefined(const Shape& shape);

/**
 * check shape fully defined or not
 * @param shape Shape is checked
 * @return whether shape is fully defined
 */
bool ShapeFullyDefined(const GeShape& shape);

/**
 * check shape known or not
 * @param shape Shape is checked
 * @return whether rank is known
 */
bool RankKnown(const Shape& shape);

/**
 * check ge_shape known or not
 * @param shape GeShape is checked
 * @return whether rank is known
 */
bool RankKnown(const GeShape& shape);

/**
 * make a unknown shape with rank
 * @return unknown shape
 */
Shape UnknownShapeOfRank(int64_t rank);

/**
 * check dim value known or not
 * @param shape which Shape need check dim value
 * @param dimIndex the index of dim
 * @return whether dim value is known
 */
bool ValueKnown(const Shape& shape, const size_t& dim_index);

/**
 * Validates the 3 component tensors of a sparse tensor
 * have the proper shapes.
 * @param sparse indices shape
 * @param sparse values shape
 * @param sparse shape
 * @return status whether this operation success
 */
graphStatus ValidateSparseTensor(const TensorDesc& indices, const TensorDesc& values, const TensorDesc& shape,
                                 const char* op_name);

/**
 * Fill op_desc with input shape
 * @param op_desc Operator desc ptr
 * @param shape input tensor shape
 * @param shape input tensor datatype
 */
void FillOpDesc(GeTensorDescPtr& op_desc, const GeShape& shape, const DataType& data_type = DT_FLOAT);

/**
 * Fill op_desc with input shape
 * @param op Operator desc ptr
 * @param shape input tensor shape
 * @param data_type input tensor datatype
 */
void FillOpDesc(TensorDesc& op_desc, const Shape& shape,
                const DataType& data_type = DT_FLOAT);

/**
 * InferShapeErrorReport info
 * @param op_name Operator name
 * @param op_type Operator type
 * @param value Operator value
 * @param reason error reason
 */
void InferShapeErrorReport(const std::string& op_name, const std::string& op_type,
                           const std::string& value, const std::string& reason);

/**
 * @brief get string from data type
 * @param dtype data type
 * @return string of data type
 */
std::string DTypeStr(DataType dtype);

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_COMMON_SHAPE_FNS_H_
