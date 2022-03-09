/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/ge_tensor.h"

#include <cstring>
#include <map>
#include <securec.h>
#include "graph/debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "graph/ge_tensor_impl.h"
#include "graph/ge_attr_value.h"
#include "graph/model_serialize.h"
#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/mem_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace{
const char *const kKeyDataTypeSelfDefined = "__tensor_desc_data_type__";
const std::map<DataType, ::ge::proto::DataType> kDataTypeMap = {
    {DT_UNDEFINED, proto::DT_UNDEFINED},
    {DT_FLOAT, proto::DT_FLOAT},
    {DT_FLOAT16, proto::DT_FLOAT16},
    {DT_INT8, proto::DT_INT8},
    {DT_UINT8, proto::DT_UINT8},
    {DT_INT16, proto::DT_INT16},
    {DT_UINT16, proto::DT_UINT16},
    {DT_INT32, proto::DT_INT32},
    {DT_INT64, proto::DT_INT64},
    {DT_UINT32, proto::DT_UINT32},
    {DT_UINT64, proto::DT_UINT64},
    {DT_BOOL, proto::DT_BOOL},
    {DT_DOUBLE, proto::DT_DOUBLE},
    {DT_DUAL, proto::DT_DUAL},
    {DT_DUAL_SUB_INT8, proto::DT_DUAL_SUB_INT8},
    {DT_DUAL_SUB_UINT8, proto::DT_DUAL_SUB_UINT8},
    {DT_COMPLEX64, proto::DT_COMPLEX64},
    {DT_COMPLEX128, proto::DT_COMPLEX128},
    {DT_QINT8, proto::DT_QINT8},
    {DT_QINT16, proto::DT_QINT16},
    {DT_QINT32, proto::DT_QINT32},
    {DT_QUINT8, proto::DT_QUINT8},
    {DT_QUINT16, proto::DT_QUINT16},
    {DT_RESOURCE, proto::DT_RESOURCE},
    {DT_STRING_REF, proto::DT_STRING_REF},
    {DT_STRING, proto::DT_STRING},
    {DT_VARIANT, proto::DT_VARIANT},
    {DT_BF16, proto::DT_BF16},
    {DT_INT4, proto::DT_INT4},
    {DT_UINT1, proto::DT_UINT1},
    {DT_INT2, proto::DT_INT2},
    {DT_UINT2, proto::DT_UINT2}
};

const std::map<DataType, int> kDataTypeSelfDefinedMap = {
    {DT_DUAL, 13},  {DT_DUAL_SUB_INT8, 14}, {DT_DUAL_SUB_UINT8, 15}, {DT_COMPLEX64, 16}, {DT_COMPLEX128, 17},
    {DT_QINT8, 18}, {DT_QINT16, 19},        {DT_QINT32, 20},         {DT_QUINT8, 21},    {DT_QUINT16, 22},
};

const std::map<DeviceType, std::string> kDeviceToStrMap = {
    {NPU, "NPU"}, {CPU, "CPU"},
};

const std::map<std::string, DeviceType> kStrToDeviceMap = {
    {"NPU", NPU}, {"CPU", CPU}
};

const string TENSOR_UTILS_SIZE = "size";
const string TENSOR_UTILS_WEIGHT_SIZE = "weight_size";
const string TENSOR_UTILS_REUSE_INPUT = "reuse_input";
const string TENSOR_UTILS_OUTPUT_TENSOR = "output_tensor";
const string TENSOR_UTILS_DEVICE_TYPE = "device_type";
const string TENSOR_UTILS_INPUT_TENSOR = "input_tensor";
const string TENSOR_UTILS_REAL_DIM_CNT = "real_dim_cnt";
const string TENSOR_UTILS_REUSE_INPUT_INDEX = "reuse_input_index";
const string TENSOR_UTILS_DATA_OFFSET = "data_offset";
const string TENSOR_UTILS_CMPS_SIZE = "cmps_size";
const string TENSOR_UTILS_CMPS_TAB = "cmps_tab";
const string TENSOR_UTILS_CMPS_TAB_OFFSET = "cmps_tab_offset";
const string TENSOR_UTILS_CMPSINFO = "cmps_info";
const string TENSOR_UTILS_ALLOFFSET_QUANTIZE_INFO = "alloffset_quantize_info";
const string TENSOR_UTILS_RC = "rc";
const string TENSOR_UTILS_ORIGIN_SHAPE = "origin_shape";
const string TENSOR_UTILS_ORIGIN_SHAPE_INITIALIZED = "origin_shape_initialized";
const string TENSOR_UTILS_ORIGIN_FORMAT = "origin_format";
const string TENSOR_UTILS_ORIGIN_DATA_TYPE = "origin_data_type";
const string TENSOR_UTILS_SHAPE_RANGE = "shape_range";
const string TENSOR_UTILS_ORIGIN_SHAPE_RANGE = "origin_shape_range";
const string TENSOR_UTILS_VALUE_RANGE = "value_range";
const string TENSOR_UTILS_REF_PORT_INDEX = "ref_port_index";
const string TENSOR_UTILS_PLACEMENT = "placement";
}

void GeTensorSerializeUtils::GeShapeAsProto(const GeShape &shape, proto::ShapeDef *proto) {
  if (proto != nullptr) {
    proto->clear_dim();
    for (auto dim : shape.GetDims()) {
      proto->add_dim(dim);
    }
  }
}
void GeTensorSerializeUtils::GeTensorDescAsProto(const GeTensorDescImpl &desc, proto::TensorDescriptor *proto) {
  if (proto != nullptr) {
    // 后续修改为从anymap中拷贝至protobuf
    if (desc.tensor_descriptor_.protoMsg_ != nullptr) {
      *proto = *(desc.tensor_descriptor_.protoMsg_);
    }

    if (!ModelSerializeImp::SerializeAllAttrsFromAnyMap(desc.attrs_.GetAllAttrs(), proto->mutable_attr())) {
      GELOGE(GRAPH_FAILED, "GeTensorDesc attr serialize failed.");
      return;
    }

    // 需要在序列化时将高频字段序列化为属性
    (*proto->mutable_attr())[TENSOR_UTILS_ORIGIN_FORMAT].set_s(TypeUtils::FormatToSerialString(desc.GetOriginFormat()));
    if (desc.GetOriginDataType() != DT_UNDEFINED) {
      (*proto->mutable_attr())[TENSOR_UTILS_ORIGIN_DATA_TYPE].set_s(
          TypeUtils::DataTypeToSerialString(desc.GetOriginDataType()));
    }

    const bool *is_origin_shape_init = desc.attrs_.GetByName<bool>(TENSOR_UTILS_ORIGIN_SHAPE_INITIALIZED);
    if (is_origin_shape_init !=  nullptr && *is_origin_shape_init) {
      auto origin_shape_proto_list = (*proto->mutable_attr())[TENSOR_UTILS_ORIGIN_SHAPE].mutable_list();
      origin_shape_proto_list->clear_i();
      for (auto dim : desc.OriginShapeReference().GetDims()) {
        origin_shape_proto_list->add_i(dim);
      }
      origin_shape_proto_list->set_val_type(proto::AttrDef::ListValue::VT_LIST_INT);
    }

    // 如果是自定义类型，此时在any map拷贝的时候已经填充了，此时设不设置set_dtype都无所谓了
    auto iter = kDataTypeMap.find(desc.GetDataType());
    if (iter != kDataTypeMap.end()) {
      proto->set_dtype(iter->second);
    } else {
      proto->set_dtype(kDataTypeMap.at(DT_UNDEFINED));
    }
    proto->set_layout(TypeUtils::FormatToSerialString(desc.GetFormat()));
    GeTensorSerializeUtils::GeShapeAsProto(desc.ShapeReference(), proto->mutable_shape());
  }
}
void GeTensorSerializeUtils::GeTensorDescAsProto(const GeTensorDesc &desc, proto::TensorDescriptor *proto) {
  GeTensorSerializeUtils::GeTensorDescAsProto(*desc.impl_, proto);
}
void GeTensorSerializeUtils::GeTensorAsProto(const GeTensorImpl &tensor, proto::TensorDef *proto) {
  if (tensor.tensor_def_.protoOwner_ != nullptr) {
    if (tensor.tensor_def_.protoMsg_ != nullptr) {
      *proto = *tensor.tensor_def_.protoMsg_;
      GeTensorDescAsProto(tensor.desc_, proto->mutable_desc());
    }
  } else {
    if (tensor.tensor_data_.impl_ != nullptr && tensor.tensor_data_.impl_->tensor_descriptor_ != nullptr) {
      GeTensorDescAsProto(*tensor.tensor_data_.impl_->tensor_descriptor_, proto->mutable_desc());
    }
    proto->set_data(tensor.tensor_data_.data(), tensor.tensor_data_.size());
  }
}
void GeTensorSerializeUtils::GeTensorAsProto(const GeTensor &tensor, proto::TensorDef *proto) {
  GeTensorSerializeUtils::GeTensorAsProto(*tensor.impl_, proto);
}

void GeTensorSerializeUtils::SetAttrToDescriptor(
    const google::protobuf::Map<std::string, ::ge::proto::AttrDef> &attr_map,
    GeIrProtoHelper<proto::TensorDescriptor> &descriptor) {
  if (descriptor.protoMsg_ == nullptr) {
    return;
  }
  auto iter = attr_map.find(TENSOR_UTILS_SIZE);
  // 下面这一大车看着是把序列化的属性上的值取出来放到成员上，哎
  if (iter != attr_map.end()) {
    descriptor.protoMsg_->set_size(iter->second.i());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_WEIGHT_SIZE))) {
    descriptor.protoMsg_->set_weight_size(iter->second.i());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_REUSE_INPUT))) {
    descriptor.protoMsg_->set_reuse_input(iter->second.b());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_OUTPUT_TENSOR))) {
    descriptor.protoMsg_->set_output_tensor(iter->second.b());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_DEVICE_TYPE))) {
    descriptor.protoMsg_->set_device_type(iter->second.s());
  } else {
    descriptor.protoMsg_->set_device_type("NPU");
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_INPUT_TENSOR))) {
    descriptor.protoMsg_->set_input_tensor(iter->second.b());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_REAL_DIM_CNT))) {
    descriptor.protoMsg_->set_real_dim_cnt(iter->second.i());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_REUSE_INPUT_INDEX))) {
    descriptor.protoMsg_->set_reuse_input_index(iter->second.i());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_DATA_OFFSET))) {
    descriptor.protoMsg_->set_data_offset(iter->second.i());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_CMPS_SIZE))) {
    descriptor.protoMsg_->set_cmps_size(iter->second.i());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_CMPS_TAB))) {
    descriptor.protoMsg_->set_cmps_tab(iter->second.s());
  }
  if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_CMPS_TAB_OFFSET))) {
    descriptor.protoMsg_->set_cmps_tab_offset(iter->second.i());
  }
}

void GeTensorSerializeUtils::AssembleGeShapeFromProto(const proto::ShapeDef *proto, GeShape &shape) {
  if (proto != nullptr) {
    shape = std::move(GeShape(nullptr, const_cast<proto::ShapeDef *>(proto)));
  }
}
void GeTensorSerializeUtils::AssembleGeTensorDescFromProto(const proto::TensorDescriptor *proto, GeTensorDesc &desc) {
  if (proto != nullptr) {
    desc = std::move(GeTensorDesc(nullptr, const_cast<proto::TensorDescriptor *>(proto)));
  }
}
void GeTensorSerializeUtils::AssembleGeTensorFromProto(const proto::TensorDef *proto, GeTensor &tensor) {
  if (proto != nullptr) {
    tensor = std::move(GeTensor(nullptr, const_cast<proto::TensorDef *>(proto)));
  }
}

class GeShapeImpl {
  using DimsType = std::vector<int64_t>;
 public:
  GeShapeImpl() = default;
  ~GeShapeImpl() = default;
  explicit GeShapeImpl(std::vector<int64_t> dims);
  GeShapeImpl(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg);

  void SetDimNum(size_t dim_num);
  void AppendDim(int64_t dim_size);
  bool IsUnknownDimNum() const;
  void SetIsUnknownDimNum();
  size_t GetDimNum() const;
  int64_t GetDim(size_t idx) const;
  graphStatus SetDim(size_t idx, int64_t value);
  std::vector<int64_t> GetDims() const;
  std::string ToString() const;
  int64_t GetShapeSize() const;
  bool IsUnknownShape() const;
  bool IsScalar() const;

  bool operator==(const GeShapeImpl &other) const;

private:
  DimsType dims_;
  friend class GeTensorDesc;
};

// Default
GeShapeImpl::GeShapeImpl(std::vector<int64_t> dims) {
  dims_.reserve(dims.size());
  for (auto dim : dims) {
    dims_.emplace_back(dim);
  }
}

void GeShapeImpl::SetDimNum(size_t dim_num) {
  dims_.resize(dim_num, UNKNOWN_DIM);
}

void GeShapeImpl::AppendDim(int64_t dim_size) {
  dims_.push_back(dim_size);
}

bool GeShapeImpl::IsUnknownDimNum() const {
  return dims_.size() == 1 && dims_[0] == UNKNOWN_DIM_NUM;
}

void GeShapeImpl::SetIsUnknownDimNum() {
  dims_.resize(1, UNKNOWN_DIM_NUM);
  dims_[0] = UNKNOWN_DIM_NUM;
}

size_t GeShapeImpl::GetDimNum() const {
  if (IsUnknownDimNum()) {
    return 0;
  }
  return dims_.size();
}

int64_t GeShapeImpl::GetDim(size_t idx) const {
  return idx < dims_.size() ? dims_[idx] : 0;
}

graphStatus GeShapeImpl::SetDim(size_t idx, int64_t value) {
  if (idx < dims_.size()) {
    dims_[idx] = value;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

std::vector<int64_t> GeShapeImpl::GetDims() const {
  std::vector<int64_t> dims;
  dims.reserve(dims_.size());
  for (auto dim : dims_) {
    dims.emplace_back(dim);
  }
  return dims;
}

std::string GeShapeImpl::ToString() const {
  if (dims_.empty()) {
    return "";
  }

  std::stringstream ss;
  ss << dims_[0];
  for (size_t i = 1; i < dims_.size(); i++) {
    ss << "," << dims_[i];
  }
  return ss.str();
}

int64_t GeShapeImpl::GetShapeSize() const {
  if (dims_.empty()) {
    return 0;
  }
  int64_t shape_size = 1;
  for (auto dim : dims_) {
    if (dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM || dim < 0) {
      return -1;
    } else if (dim == 0) {
      return 0;
    } else {
      if (shape_size > INT64_MAX / dim) {
        return -1;
      }
      shape_size *= dim;
    }
  }
  return shape_size;
}

bool GeShapeImpl::IsUnknownShape() const {
  for (auto dim : dims_) {
    if (dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM || dim < 0) {
      return true;
    }
  }
  return false;
}

bool GeShapeImpl::IsScalar() const {
  return dims_.empty();
}

GeShapeImpl::GeShapeImpl(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg) {
  if (proto_msg != nullptr) {
    for (auto &dim : *proto_msg->mutable_dim()) {
      dims_.emplace_back(dim);
    }
  }
}

bool GeShapeImpl::operator==(const GeShapeImpl &other) const {
  return this->GetDims() == other.GetDims();
}

GeShape::GeShape() : impl_(std::shared_ptr<GeShapeImpl>(new GeShapeImpl())) {}
GeShape::GeShape(std::vector<int64_t> s)
    : impl_(std::shared_ptr<GeShapeImpl>(new GeShapeImpl(std::move(s)))) {}
GeShape::GeShape(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg)
    : impl_(std::shared_ptr<GeShapeImpl>(new GeShapeImpl(proto_owner, proto_msg))) {}

GeShape::GeShape(const GeShape &other)
    : impl_(std::shared_ptr<GeShapeImpl>(new GeShapeImpl(*(other.impl_)))) {}

GeShape::GeShape(GeShape &&other)
    : impl_(std::shared_ptr<GeShapeImpl>(new GeShapeImpl(std::move(*(other.impl_))))) {}

GeShape::~GeShape() = default;

size_t GeShape::GetDimNum() const {
  return impl_->GetDimNum();
}

void GeShape::SetDimNum(size_t dim_num) {
  impl_->SetDimNum(dim_num);
}

void GeShape::AppendDim(int64_t dim_size) {
  impl_->AppendDim(dim_size);
}

bool GeShape::IsUnknownDimNum() const {
  return impl_->IsUnknownDimNum();
}

void GeShape::SetIsUnknownDimNum() {
  impl_->SetIsUnknownDimNum();
}

int64_t GeShape::GetDim(size_t idx) const {
  return impl_->GetDim(idx);
}

graphStatus GeShape::SetDim(size_t idx, int64_t value) {
  return impl_->SetDim(idx, value);
}

std::vector<int64_t> GeShape::GetDims() const {
  return impl_->GetDims();
}

std::string GeShape::ToString() const {
  return impl_->ToString();
}

int64_t GeShape::GetShapeSize() const {
  return impl_->GetShapeSize();
}

///
/// @brief Check is unknown shape
/// @return bool
/// ///
bool GeShape::IsUnknownShape() const {
  return impl_->IsUnknownShape();
}

///
/// @brief Check is a scalar
/// @return bool
///
bool GeShape::IsScalar() const {
  return impl_->IsScalar();
}

GeShape &GeShape::operator=(const GeShape &other) {
  if (&other != this) {
    *impl_ = *(other.impl_);
  }
  return *this;
}

GeShape &GeShape::operator=(GeShape &&other) {
  if (&other != this) {
    *impl_ = std::move(*(other.impl_));
  }
  return *this;
}

bool GeShape::operator==(const GeShape &other) const {
  return *impl_ == *(other.impl_);
}

GeTensorDescImpl::GeTensorDescImpl() {
  tensor_descriptor_.InitDefault();
  Init();
}

GeTensorDescImpl::GeTensorDescImpl(const GeShape &shape, Format format, DataType dt) : GeTensorDescImpl() {
  SetFormat(format);
  SetDataType(dt);
  shape_ = shape;
}

GeTensorDescImpl::GeTensorDescImpl(const GeTensorDescImpl &desc) : GeTensorDescImpl() {
  // 替换为any map后删除该函数
  tensor_descriptor_.CopyValueFrom(desc.tensor_descriptor_);
  shape_ = desc.shape_;
  format_ = desc.format_;
  dtype_ = desc.dtype_;
  origin_shape_ = desc.origin_shape_;
  origin_format_ = desc.origin_format_;
  origin_dtype_ = desc.origin_dtype_;
  attrs_ = desc.attrs_;
}

GeTensorDescImpl::GeTensorDescImpl(GeTensorDescImpl &&desc) : GeTensorDescImpl() {
  // 替换为any map后删除该函数
  tensor_descriptor_.MoveValueFrom(std::move(desc.tensor_descriptor_));
  shape_ = std::move(desc.shape_);
  format_ = desc.format_;
  dtype_ = desc.dtype_;
  origin_shape_ = std::move(desc.origin_shape_);
  origin_format_ = desc.origin_format_;
  origin_dtype_ = desc.origin_dtype_;
  attrs_ = std::move(desc.attrs_);
}


GeTensorDescImpl::GeTensorDescImpl(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg)
    : GeTensorDescImpl() {
  // 替换为any map后删除该函数
  if (tensor_descriptor_.protoMsg_ != nullptr && proto_msg != nullptr) {
    // 后续修改为从protobuf中拷贝至anymap
    *tensor_descriptor_.protoMsg_ = *proto_msg;
    auto &attr_map = *(proto_msg->mutable_attr());
    auto iter = attr_map.find(TENSOR_UTILS_ORIGIN_FORMAT);
    // 先将高频字段从protobuf中恢复
    if (iter != attr_map.end()) {
      origin_format_ = TypeUtils::SerialStringToFormat(iter->second.s());
    }
    if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_ORIGIN_DATA_TYPE))) {
      origin_dtype_ = TypeUtils::SerialStringToDataType(iter->second.s());
    }
    if (attr_map.end() != (iter = attr_map.find(TENSOR_UTILS_ORIGIN_SHAPE))) {
      origin_shape_.SetDimNum(iter->second.list().i_size());
      size_t i = 0;
      for (auto dim : iter->second.list().i()) {
        origin_shape_.SetDim(i++, dim);
      }
    }

    GeTensorSerializeUtils::SetAttrToDescriptor(attr_map, tensor_descriptor_);

    dtype_ = DT_UNDEFINED;
    auto it_data_type = attr_map.find(kKeyDataTypeSelfDefined);
    if (it_data_type == attr_map.end()) {
      auto proto_dtype = proto_msg->dtype();
      for (auto item : kDataTypeMap) {
        if (item.second == proto_dtype) {
          dtype_ = item.first;
        }
      }
    } else { // Custom defined data type set
      int64_t data_type_proto = it_data_type->second.i();
      for (auto it : kDataTypeSelfDefinedMap) {
        if (it.second == data_type_proto) {
          dtype_ = it.first;
        }
      }
    }

    format_ = TypeUtils::SerialStringToFormat(proto_msg->layout());

    auto dim_size = proto_msg->shape().dim_size();
    if (dim_size > 0) {
      shape_.SetDimNum(dim_size);
      auto &proto_dims = proto_msg->shape().dim();
      size_t i = 0;
      for (auto dim : proto_dims) {
        (void)shape_.SetDim(i++, dim);
      }
    }
  }
}

void GeTensorDescImpl::SetDataType(DataType dtype) {
  dtype_ = dtype;
  return;
  // 原始的逻辑似乎是在表达，先在原始支持类型kDataTypeMap中找，如果找到了，就认为是基本类型，删除自定义类型属性
  // 如果kDataTypeMap中没找到，则尝试在kDataTypeSelfDefinedMap中找，如果找到了，设置到自定义类型属性上
  // 后续修改为对Any map的操作，即如果是常规类型，删除kKeyDataTypeSelfDefined，否则设置kKeyDataTypeSelfDefined为自定义枚举
  if (tensor_descriptor_.protoMsg_ != nullptr) {
    auto iter_basic_type = kDataTypeMap.find(dtype);
    if (iter_basic_type != kDataTypeMap.end()) {
      (void)tensor_descriptor_.protoMsg_->mutable_attr()->erase(kKeyDataTypeSelfDefined);
    } else {
      auto iter_custom_type = kDataTypeSelfDefinedMap.find(dtype);
      if (iter_custom_type != kDataTypeSelfDefinedMap.end()) {
        (*tensor_descriptor_.protoMsg_->mutable_attr())[kKeyDataTypeSelfDefined].set_i(iter_custom_type->second);
      }
    }
  }
}

void GeTensorDescImpl::SetOriginDataType(DataType dtype) {
  origin_dtype_ = dtype;
}

DataType GeTensorDescImpl::GetOriginDataType() const {
  return origin_dtype_;
}

void GeTensorDescImpl::Init() {
  SetFormat(FORMAT_ND);
  SetDataType(DT_FLOAT);
  SetOriginFormat(FORMAT_ND);
  SetOriginDataType(DT_UNDEFINED);
  SetDeviceType(DeviceType::NPU);
  if (tensor_descriptor_.GetProtoMsg() == nullptr) {
    REPORT_CALL_ERROR("E19999", "ProtoType is nullptr.");
    GELOGE(GRAPH_FAILED, "[Get][ProtoMsg] ProtoType nullptr.");
    return;
  }
  tensor_descriptor_.GetProtoMsg()->set_has_out_attr(true);
}

void GeTensorDescImpl::SetFormat(Format format) {
  format_ = format;
}

void GeTensorDescImpl::SetOriginFormat(Format format) {
  origin_format_ = format;
}

Format GeTensorDescImpl::GetOriginFormat() const {
  return origin_format_;
}

GeShape &GeTensorDescImpl::ShapeReference() const {
  return shape_;
}

GeShape &GeTensorDescImpl::OriginShapeReference() const {
  return origin_shape_;
}

bool GeTensorDescImpl::GeTensorDescAttrsAreEqual(const GeTensorDescImpl &r_ge_tensor_desc) const {
  const auto &tensor_descriptor = this->tensor_descriptor_.GetProtoMsg();
  const auto &r_tensor_descriptor = r_ge_tensor_desc.tensor_descriptor_.GetProtoMsg();
  if (shape_.ToString() != r_ge_tensor_desc.shape_.ToString() ||
      dtype_ != r_ge_tensor_desc.dtype_ ||
      format_ != r_ge_tensor_desc.format_) {
    return false;
  }
  if ((tensor_descriptor != nullptr) && (r_tensor_descriptor != nullptr)) {
    // Message TensorDescriptor in ge_ir.proto
    return (IsEqual(tensor_descriptor->name(), r_tensor_descriptor->name(), "TensorDescriptor.name()") &&
            IsEqual(tensor_descriptor->has_out_attr(), r_tensor_descriptor->has_out_attr(),
                    "TensorDescriptor.has_out_attr()") &&
            IsEqual(tensor_descriptor->size(), r_tensor_descriptor->size(), "TensorDescriptor.size()") &&
            IsEqual(tensor_descriptor->weight_size(), r_tensor_descriptor->weight_size(),
                    "TensorDescriptor.weight_size()") &&
            IsEqual(tensor_descriptor->reuse_input(), r_tensor_descriptor->reuse_input(),
                    "TensorDescriptor.reuse_input()") &&
            IsEqual(tensor_descriptor->output_tensor(), r_tensor_descriptor->output_tensor(),
                    "TensorDescriptor.output_tensor()") &&
            IsEqual(tensor_descriptor->device_type(), r_tensor_descriptor->device_type(),
                    "TensorDescriptor.device_type()") &&
            IsEqual(tensor_descriptor->input_tensor(), r_tensor_descriptor->input_tensor(),
                    "TensorDescriptor.input_tensor()") &&
            IsEqual(tensor_descriptor->real_dim_cnt(), r_tensor_descriptor->real_dim_cnt(),
                    "TensorDescriptor.real_dim_cnt()") &&
            IsEqual(tensor_descriptor->reuse_input_index(), r_tensor_descriptor->reuse_input_index(),
                    "TensorDescriptor.reuse_input_index()") &&
            IsEqual(tensor_descriptor->data_offset(), r_tensor_descriptor->data_offset(),
                    "TensorDescriptor.data_offset()") &&
            IsEqual(tensor_descriptor->cmps_size(), r_tensor_descriptor->cmps_size(), "TensorDescriptor.cmps_size()") &&
            IsEqual(tensor_descriptor->cmps_tab(), r_tensor_descriptor->cmps_tab(), "TensorDescriptor.cmps_tab()") &&
            IsEqual(tensor_descriptor->cmps_tab_offset(), r_tensor_descriptor->cmps_tab_offset(),
                    "TensorDescriptor.cmps_tab_offset()"));
  } else {
    return ((tensor_descriptor == nullptr) && (r_tensor_descriptor == nullptr));
  }
}

bool GeTensorDescImpl::operator==(const GeTensorDescImpl &r_ge_tensor_desc) const {
  return (shape_ == r_ge_tensor_desc.shape_ && origin_shape_ == r_ge_tensor_desc.origin_shape_ &&
          format_ == r_ge_tensor_desc.format_ && origin_format_ == r_ge_tensor_desc.origin_format_ &&
          dtype_ == r_ge_tensor_desc.dtype_ && origin_dtype_ == r_ge_tensor_desc.origin_dtype_ &&
          GeTensorDescAttrsAreEqual(r_ge_tensor_desc));
}

ProtoAttrMap &GeTensorDescImpl::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &GeTensorDescImpl::GetAttrMap() const {
  return attrs_;
}

void GeTensorDescImpl::SetShape(const GeShape &shape) { ShapeReference() = std::move(shape); }

Format GeTensorDescImpl::GetFormat() const {
  return format_;
}

void GeTensorDescImpl::SetDeviceType(DeviceType type) {
  auto iter = kDeviceToStrMap.find(type);
  std::string type_str;
  if (iter != kDeviceToStrMap.end()) {
    type_str = iter->second;
  } else {
    GELOGW("[Set][DeviceType] not found device type.");
  }
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_device_type(type_str);
  }
}

void GeTensorDescImpl::SetName(const std::string &name) {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    tensor_descriptor_msg->set_name(name);
    return;
  }
  GELOGW("[SetName]tensor_descriptor_msg is null.");
}

const std::string GeTensorDescImpl::GetName() const {
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg != nullptr) {
    return tensor_descriptor_msg->name();
  }
  GELOGW("[GetName]tensor_descriptor_msg is null.");
  return "";
}

DataType GeTensorDescImpl::GetDataType() const {
  return dtype_;
  // 下面仅在当前的自定义类型逻辑确实需要的时候才添加。
  // 后续变为判断any map是否为空
  auto tensor_descriptor_msg = tensor_descriptor_.GetProtoMsg();
  if (tensor_descriptor_msg == nullptr) {
    auto &attr_map = *(tensor_descriptor_msg->mutable_attr());
    auto it_data_type = attr_map.find(kKeyDataTypeSelfDefined);
    if (it_data_type == attr_map.end()) {
      return dtype_;
    } else { // Custom defined data type set
      int64_t data_type_proto = it_data_type->second.i();
      for (auto it : kDataTypeSelfDefinedMap) {
        if (it.second == data_type_proto) {
          return it.first;
        }
      }
    }
  }
  return dtype_;
}

GeTensorDescImpl &GeTensorDescImpl::operator=(const GeTensorDescImpl &desc) {
  // 替换为any map后删除该函数
  if (&desc != this) {
    tensor_descriptor_.CopyValueFrom(desc.tensor_descriptor_);
    shape_ = desc.shape_;
    format_ = desc.format_;
    dtype_ = desc.dtype_;
    origin_shape_ = desc.origin_shape_;
    origin_format_ = desc.origin_format_;
    origin_dtype_ = desc.origin_dtype_;
    attrs_ = desc.attrs_;
  }
  return *this;
}

GeTensorDescImpl &GeTensorDescImpl::operator=(GeTensorDescImpl &&desc) {
  // 替换为any map后删除该函数
  if (&desc != this) {
    tensor_descriptor_.CopyValueFrom(std::move(desc.tensor_descriptor_));
    shape_ = std::move(desc.shape_);
    format_ = desc.format_;
    dtype_ = desc.dtype_;
    origin_shape_ = std::move(desc.origin_shape_);
    origin_format_ = desc.origin_format_;
    origin_dtype_ = desc.origin_dtype_;
    attrs_ = std::move(desc.attrs_);
  }
  return *this;
}

GeTensorDesc::GeTensorDesc()
    : impl_(ComGraphMakeShared<GeTensorDescImpl>()) {}

// Default
GeTensorDesc::GeTensorDesc(const GeShape &shape, Format format, DataType dt)
    : impl_(ComGraphMakeShared<GeTensorDescImpl>(shape, format, dt)) {}

// Default
GeTensorDesc::GeTensorDesc(const GeTensorDesc &desc)
    : AttrHolder(desc),
      impl_(ComGraphMakeShared<GeTensorDescImpl>(*(desc.impl_))) {}

// Default
GeTensorDesc::GeTensorDesc(GeTensorDesc &&desc)
    : AttrHolder(std::move(desc)),
      impl_(ComGraphMakeShared<GeTensorDescImpl>(std::move(*(desc.impl_)))) {}

GeTensorDesc::~GeTensorDesc() = default;

GeTensorDesc::GeTensorDesc(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg)
    : impl_(ComGraphMakeShared<GeTensorDescImpl>(proto_owner, proto_msg)) {
  if (proto_msg != nullptr) {
    if (!ModelSerializeImp::DeserializeAllAttrsToAttrHolder(proto_msg->attr(), this)) {
      GELOGW("GeTensorDesc attr deserialize failed.");
    }
  }
}

bool GeTensorDesc::GeTensorDescAttrsAreEqual(const GeTensorDesc &r_ge_tensor_desc) const {
  return impl_->GeTensorDescAttrsAreEqual(*(r_ge_tensor_desc.impl_));
}

bool GeTensorDesc::operator==(const GeTensorDesc &r_ge_tensor_desc) const {
  return *impl_ == *r_ge_tensor_desc.impl_;
}

GeShape &GeTensorDesc::ShapeReference() const {
  return impl_->ShapeReference();
}

void GeTensorDesc::RefTo(const GeTensorDesc &tensorDesc) {
  impl_->RefTo(*(tensorDesc.impl_));
}

ProtoAttrMap &GeTensorDesc::MutableAttrMap() {
  return impl_->MutableAttrMap();
}

ConstProtoAttrMap &GeTensorDesc::GetAttrMap() const {
  return impl_->GetAttrMap();
}

void GeTensorDesc::Update(const GeShape &shape, Format format, DataType dt) {
  ShapeReference() = shape;
  SetFormat(format);
  SetDataType(dt);
}
const GeShape &GeTensorDesc::GetShape() const { return ShapeReference(); }

GeShape &GeTensorDesc::MutableShape() { return ShapeReference(); }

void GeTensorDesc::SetShape(const GeShape &shape) { ShapeReference() = shape; }

void GeTensorDesc::SetShape(GeShape &&shape) { ShapeReference() = std::move(shape); }

// set shape with -2, it stand for unknown shape
void GeTensorDesc::SetUnknownDimNumShape() { SetShape(GeShape({UNKNOWN_DIM_NUM})); }

// for unknown shape
graphStatus GeTensorDesc::SetValueRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  std::vector<vector<int64_t>> value_range;
  for (const auto &ele : range) {
    value_range.emplace_back(std::vector<int64_t>({ele.first, ele.second}));
  }
  auto ret = AttrUtils::SetListListInt(this, TENSOR_UTILS_VALUE_RANGE, value_range);
  return ret ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus GeTensorDesc::GetValueRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  std::vector<vector<int64_t>> value_range;
  (void) AttrUtils::GetListListInt(this, TENSOR_UTILS_VALUE_RANGE, value_range);

  for (const auto &ele : value_range) {
    // here must be only two elemenet because pair
    if (ele.size() != 2) {
      REPORT_INNER_ERROR("E19999", "value_range must contain only 2 value but really is %zu", ele.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] value_range must contain only 2 value but really is %zu", ele.size());
      return GRAPH_FAILED;
    }
    range.emplace_back(std::make_pair(ele[0], ele[1]));
  }

  return GRAPH_SUCCESS;
}

graphStatus GeTensorDesc::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  std::vector<vector<int64_t>> shape_range;
  for (const auto &ele : range) {
    shape_range.emplace_back(std::vector<int64_t>({ele.first, ele.second}));
  }
  auto ret = AttrUtils::SetListListInt(this, TENSOR_UTILS_SHAPE_RANGE, shape_range);
  return ret ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus GeTensorDesc::SetOriginShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  std::vector<vector<int64_t>> origin_shape_range;
  for (const auto &ele : range) {
    origin_shape_range.emplace_back(std::vector<int64_t>({ele.first, ele.second}));
  }
  auto ret = AttrUtils::SetListListInt(this, TENSOR_UTILS_ORIGIN_SHAPE_RANGE, origin_shape_range);
  return ret ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus GeTensorDesc::GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  std::vector<vector<int64_t>> shape_range;
  (void)AttrUtils::GetListListInt(this, TENSOR_UTILS_SHAPE_RANGE, shape_range);

  for (const auto &ele : shape_range) {
    // here must be only two elemenet because pair
    if (ele.size() != 2) {
      REPORT_INNER_ERROR("E19999", "shape_range must contain only 2 value but really is %zu", ele.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] shape_range must contain only 2 value but really is %zu", ele.size());
      return GRAPH_FAILED;
    }
    std::pair<int64_t, int64_t> pair({ele[0], ele[1]});
    range.emplace_back(pair);
  }

  return GRAPH_SUCCESS;
}

graphStatus GeTensorDesc::GetOriginShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  std::vector<vector<int64_t>> origin_shape_range;
  (void)AttrUtils::GetListListInt(this, TENSOR_UTILS_ORIGIN_SHAPE_RANGE, origin_shape_range);

  for (const auto &ele : origin_shape_range) {
    // here must be only two elemenet because pair
    if (ele.size() != 2) {
      REPORT_INNER_ERROR("E19999", "origin_shape_range must contain only 2 value but really is %zu", ele.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] origin_shape_range must contain only 2 value but really is %zu", ele.size());
      return GRAPH_FAILED;
    }
    std::pair<int64_t, int64_t> pair({ele[0], ele[1]});
    range.emplace_back(pair);
  }

  return GRAPH_SUCCESS;
}

const GeShape &GeTensorDesc::GetOriginShape() const {
  return impl_->OriginShapeReference();
}

void GeTensorDesc::SetOriginShape(const GeShape &origin_shape) {
  impl_->OriginShapeReference() = origin_shape;
  (void)AttrUtils::SetBool(this, TENSOR_UTILS_ORIGIN_SHAPE_INITIALIZED, true);
}

bool GeTensorDesc::IsOriginShapeInitialized() const {
  bool original_shape_initialized = false;
  (void)AttrUtils::GetBool(this, TENSOR_UTILS_ORIGIN_SHAPE_INITIALIZED, original_shape_initialized);
  return original_shape_initialized;
}

Format GeTensorDesc::GetFormat() const {
  return impl_->GetFormat();
}

void GeTensorDesc::SetFormat(Format format) {
  return impl_->SetFormat(format);
}

void GeTensorDesc::SetName(const std::string &name) {
  return impl_->SetName(name);
}

const std::string GeTensorDesc::GetName() const {
  return impl_->GetName();
}

Format GeTensorDesc::GetOriginFormat() const {
  return impl_->GetOriginFormat();
}

void GeTensorDesc::SetOriginFormat(Format origin_format) {
  impl_->SetOriginFormat(origin_format);
}

void GeTensorDesc::SetDataType(DataType dataType) {
  return impl_->SetDataType(dataType);
}

DataType GeTensorDesc::GetDataType() const {
  return impl_->GetDataType();
}

void GeTensorDesc::SetOriginDataType(DataType origin_data_type) {
  impl_->SetOriginDataType(origin_data_type);
}

DataType GeTensorDesc::GetOriginDataType() const {
  return impl_->GetOriginDataType();
}

std::vector<uint32_t> GeTensorDesc::GetRefPortIndex() const {
  vector<uint32_t> ref_port_index;
  (void)AttrUtils::GetListInt(this, TENSOR_UTILS_REF_PORT_INDEX, ref_port_index);
  return ref_port_index;
}

void GeTensorDesc::SetRefPortByIndex(const std::vector<uint32_t> &index) {
  (void)AttrUtils::SetListInt(this, TENSOR_UTILS_REF_PORT_INDEX, index);
}

Placement GeTensorDesc::GetPlacement() const {
  int64_t placement = 0;
  (void)AttrUtils::GetInt(this, TENSOR_UTILS_PLACEMENT, placement);
  return static_cast<Placement>(placement);
}

void GeTensorDesc::SetPlacement(Placement placement) {
  (void)AttrUtils::SetInt(this, TENSOR_UTILS_PLACEMENT, static_cast<int64_t>(placement));
}

graphStatus GeTensorDesc::IsValid() const {
  auto dtype = this->GetDataType();
  auto format = this->GetFormat();
  if (dtype == DT_UNDEFINED && format == FORMAT_RESERVED) {
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

GeTensorDesc GeTensorDesc::Clone() const { return *this; }

GeTensorDesc &GeTensorDesc::operator=(const GeTensorDesc &desc) {
  if (&desc != this) {
    AttrHolder::CopyFrom(desc);
    *impl_ = *(desc.impl_);
  }
  return *this;
}

GeTensorDesc &GeTensorDesc::operator=(GeTensorDesc &&desc) {
  if (&desc != this) {
    AttrHolder::CopyFrom(desc);
    *impl_ = std::move(*(desc.impl_));
  }
  return *this;
}

uint32_t TensorDataImpl::invalid_data_ = 0x3A2D2900;

TensorDataImpl::TensorDataImpl(const TensorDataImpl &other) {
  // Share data
  tensor_descriptor_ = other.tensor_descriptor_;
  aligned_ptr_ = other.aligned_ptr_;
  length_ = other.length_;
}

TensorDataImpl &TensorDataImpl::operator=(const TensorDataImpl &other) {
  if (&other != this) {
    // Share data
    tensor_descriptor_ = other.tensor_descriptor_;
    aligned_ptr_ = other.aligned_ptr_;
    length_ = other.length_;
  }
  return *this;
}

graphStatus TensorDataImpl::SetData(const uint8_t *data, size_t size) {
  if (size == 0) {
    GELOGI("size is 0");
    clear();
    return GRAPH_SUCCESS;
  }
  if (data == nullptr) {
    GELOGI("data addr is empty");
    return GRAPH_SUCCESS;
  }

  if (MallocAlignedPtr(size) == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[Malloc][Memory] failed, size=%zu", size);
    return GRAPH_FAILED;
  }

  size_t remain_size = size;
  auto dst_addr = reinterpret_cast<uintptr_t>(aligned_ptr_->MutableGet());
  auto src_addr = reinterpret_cast<uintptr_t>(data);
  while (remain_size > SECUREC_MEM_MAX_LEN) {
    if (memcpy_s(reinterpret_cast<void *>(dst_addr), SECUREC_MEM_MAX_LEN,
                 reinterpret_cast<const void *>(src_addr), SECUREC_MEM_MAX_LEN) != EOK) {
      REPORT_CALL_ERROR("E19999", "memcpy failed, size = %lu", SECUREC_MEM_MAX_LEN);
      GELOGE(INTERNAL_ERROR, "[Memcpy][Data] failed, size = %lu", SECUREC_MEM_MAX_LEN);
      return GRAPH_FAILED;
    }
    remain_size -= SECUREC_MEM_MAX_LEN;
    dst_addr += SECUREC_MEM_MAX_LEN;
    src_addr += SECUREC_MEM_MAX_LEN;
  }
  if (memcpy_s(reinterpret_cast<void *>(dst_addr), remain_size,
               reinterpret_cast<const void *>(src_addr), remain_size) != EOK) {
    REPORT_CALL_ERROR("E19999", "memcpy failed, size=%zu", remain_size);
    GELOGE(INTERNAL_ERROR, "[Memcpy][Data] failed, size=%zu", remain_size);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void TensorDataImpl::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, size_t size) {
  aligned_ptr_ = std::move(aligned_ptr);
  length_ = size;
}

graphStatus TensorDataImpl::SetData(uint8_t *data, size_t size, const AlignedPtr::Deleter &delete_fuc) {
  if (size == 0) {
    GELOGW("[Set][Data] Input size is 0");
    clear();
    return GRAPH_SUCCESS;
  }
  if (data == nullptr) {
    REPORT_CALL_ERROR("E19999", "data is nullptr");
    GELOGE(GRAPH_FAILED, "[Check][Param] data is nullptr");
    return GRAPH_FAILED;
  }
  length_ = size;
  aligned_ptr_ = AlignedPtr::BuildFromData(data, delete_fuc);
  return GRAPH_SUCCESS;
}

const uint8_t *TensorDataImpl::MallocAlignedPtr(size_t size) {
  if (size == 0) {
    GELOGW("[Check][Param] Input data size is 0");
    clear();
    return reinterpret_cast<const uint8_t *>(&invalid_data_);
  }
  if (length_ != size) {
    aligned_ptr_.reset();
  }
  length_ = size;
  if (aligned_ptr_ == nullptr) {
    aligned_ptr_ = MakeShared<AlignedPtr>(length_);
    if (aligned_ptr_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "create AlignedPtr failed.");
      GELOGE(INTERNAL_ERROR, "[Create][AlignedPtr] failed.");
      return nullptr;
    }
  }

  return aligned_ptr_->Get();
}

size_t TensorDataImpl::GetSize() const { return length_; }

const uint8_t *TensorDataImpl::GetData() const {
  if (length_ == 0) {
    return reinterpret_cast<const uint8_t *>(&invalid_data_);
  }
  if (aligned_ptr_ == nullptr) {
    return nullptr;
  }
  return aligned_ptr_->Get();
}

uint8_t *TensorDataImpl::GetData() {
  if (length_ == 0) {
    return reinterpret_cast<uint8_t *>(&invalid_data_);
  }
  if (aligned_ptr_ == nullptr) {
    return nullptr;
  }
  return aligned_ptr_->MutableGet();
}

void TensorDataImpl::clear() {
  aligned_ptr_.reset();
  length_ = 0;
}

uint8_t TensorDataImpl::operator[](size_t index) const {
  if (aligned_ptr_ != nullptr && index < length_) {
    return *(aligned_ptr_->MutableGet() + index);
  }
  return 0xff;
}

TensorData::TensorData()
    : impl_(std::shared_ptr<TensorDataImpl>(new TensorDataImpl())) {}

TensorData::TensorData(const TensorData &other)
    : impl_(std::shared_ptr<TensorDataImpl>(new TensorDataImpl(*(other.impl_)))) {}

TensorData::~TensorData() = default;

TensorData &TensorData::operator=(const TensorData &other) {
  if (&other != this) {
    *impl_ = *(other.impl_);
  }
  return *this;
}

graphStatus TensorData::SetData(std::vector<uint8_t> &&data) { return SetData(data.data(), data.size()); }
graphStatus TensorData::SetData(const std::vector<uint8_t> &data) { return SetData(data.data(), data.size()); }
graphStatus TensorData::SetData(const Buffer &data) { return SetData(data.data(), data.size()); }
graphStatus TensorData::SetData(const TensorData &data) { return SetData(data.data(), data.size()); }

graphStatus TensorData::SetData(const uint8_t *data, size_t size) {
  return impl_->SetData(data, size);
}

graphStatus TensorData::SetData(uint8_t *data, size_t size, const AlignedPtr::Deleter &delete_fuc) {
  return impl_->SetData(data, size, delete_fuc);
}

void TensorData::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, size_t size) {
  impl_->SetData(aligned_ptr, size);
}

const uint8_t *TensorData::MallocAlignedPtr(size_t size) {
  return impl_->MallocAlignedPtr(size);
}

size_t TensorData::GetSize() const {
  return impl_->GetSize();
}

const uint8_t *TensorData::GetData() const {
  return impl_->GetData();
}

uint8_t *TensorData::GetData() {
  return impl_->GetData();
}

const std::uint8_t *TensorData::data() const { return GetData(); }
std::uint8_t *TensorData::data() { return GetData(); }
std::size_t TensorData::size() const { return GetSize(); }
void TensorData::clear() {
  impl_->clear();
}

uint8_t TensorData::operator[](size_t index) const {
  return (*impl_)[index];
}

const std::shared_ptr<AlignedPtr> &TensorData::GetAlignedPtr() {
  return impl_->GetAlignedPtr();
}

GeTensorImpl::GeTensorImpl() : tensor_def_(nullptr, nullptr), desc_(), tensor_data_()  {
  if (desc_.impl_ != nullptr) {
    if (tensor_data_.impl_ != nullptr) {
      // 这里修改了实现
      tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
    }
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc) : GeTensorImpl() {
  DescReference() = tensor_desc;
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, const vector<uint8_t> &data) : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (tensor_data_.SetData(data) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, const uint8_t *data, size_t size) : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (tensor_data_.SetData(data, size) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(GeTensorDesc &&tensor_desc, vector<uint8_t> &&data) : GeTensorImpl() {
  DescReference() = std::move(tensor_desc);
  if (tensor_data_.SetData(data) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, const Buffer &data) : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (data.size() == 0) {
    GELOGI("GetSize res is 0.");
  }
  if (data.data() == nullptr) {
    GELOGI("data addr is null.");
  }
  if (tensor_data_.SetData(data) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr, size_t size)
    : GeTensorImpl() {
  DescReference() = tensor_desc;
  tensor_data_.SetData(std::move(aligned_ptr), size);
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, size_t size) : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (tensor_data_.MallocAlignedPtr(size) == nullptr) {
    GELOGW("[Malloc][Memory] Malloc memory failed, size=%zu", size);
  }
}

GeTensorImpl::GeTensorImpl(const ProtoMsgOwner &proto_owner, proto::TensorDef *proto_msg)
    : tensor_def_(proto_owner, proto_msg) {
  // 这里后续改为反序列化接口调用，从proto恢复GeTensorDesc
  desc_ = GeTensorDesc(proto_owner, proto_msg == nullptr ? nullptr : proto_msg->mutable_desc());
  tensor_data_ = TensorData();
  if (tensor_data_.impl_ != nullptr && desc_.impl_ != nullptr) {
    // 之前没有把TensorData上的proto变为GeTensorDesc，因为TensorData创建后不会修改，多个TensorData通过GeIrProto共享
    // 但是！原本的语义是TensorData上的proto::TensorDescriptor与Tensor上的GeTensorDesc是共享的，当GeTensorDesc改造完
    // 这种共享的能力就消失了，这会导致在GeTensor创建后，对GeTensorDesc的修改无法反应到TensorData上，看起来只能将TensorData
    // 上的proto::TensorDescriptor修改为GeTensorDescImpl，并且需要与GeTensor的GeTensorDesc共享
    tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
  }

  if (proto_msg != nullptr) {
    if (proto_owner != nullptr) {
      BuildAlignerPtrWithProtoData();
    } else {
      (void)tensor_data_.SetData(reinterpret_cast<const uint8_t *>(proto_msg->data().data()), proto_msg->data().size());
    }
  }
}

GeTensorDesc &GeTensorImpl::DescReference() const {
  return desc_;
}

void GeTensorImpl::BuildAlignerPtrWithProtoData() {
  auto proto_msg = tensor_def_.GetProtoMsg();
  if ((proto_msg == nullptr) || (reinterpret_cast<const uint8_t *>(proto_msg->data().data()) == tensor_data_.data())) {
    return;
  }
  if (tensor_data_.impl_ == nullptr) {
    return;
  }

  tensor_data_.impl_->length_ = proto_msg->data().size();
  tensor_data_.impl_->aligned_ptr_.reset();
  tensor_data_.impl_->aligned_ptr_ =
      AlignedPtr::BuildFromAllocFunc([&proto_msg](std::unique_ptr<uint8_t[], AlignedPtr::Deleter> &ptr) {
                                       ptr.reset(const_cast<uint8_t *>(
                                           reinterpret_cast<const uint8_t *>(proto_msg->data().data())));
                                     },
                                     [](uint8_t *ptr) {
                                       ptr = nullptr;
                                     });
}

graphStatus GeTensorImpl::SetData(vector<uint8_t> &&data) {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    proto_msg->set_data(data.data(), data.size());
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data);
}

graphStatus GeTensorImpl::SetData(const vector<uint8_t> &data) {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    proto_msg->set_data(data.data(), data.size());
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data);
}

graphStatus GeTensorImpl::SetData(const uint8_t *data, size_t size) {
  if (size > 0) {
    GE_CHECK_NOTNULL(data);
  }
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    proto_msg->set_data(data, size);
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data, size);
}

graphStatus GeTensorImpl::SetData(const Buffer &data) {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    if (data.size() == 0) {
      GELOGI("GetSize res is 0.");
    }
    if (data.data() == nullptr) {
      GELOGI("data addr is null.");
    }
    proto_msg->set_data(data.data(), data.size());
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data);
}

graphStatus GeTensorImpl::SetData(const TensorData &data) {
  return SetData(data.data(), data.size());
}

graphStatus GeTensorImpl::SetData(uint8_t *data, size_t size, const AlignedPtr::Deleter &delete_fuc) {
  return tensor_data_.SetData(data, size, delete_fuc);
}

void GeTensorImpl::ClearData() {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto proto_msg = tensor_def_.GetProtoMsg();
    if (proto_msg != nullptr) {
      proto_msg->clear_data();
    }
  }
  tensor_data_.clear();
}

void GeTensorImpl::Clone(GeTensorImpl &tensor) const {
  if (tensor.desc_.impl_ != nullptr && desc_.impl_ != nullptr) {
    *(tensor.desc_.impl_) = *(desc_.impl_);
  }
  if (tensor.tensor_data_.impl_ != nullptr && tensor.desc_.impl_ != nullptr) {
    tensor.tensor_data_.impl_->tensor_descriptor_ = tensor.desc_.impl_;
  }
  tensor.SetData(GetData());
}

std::shared_ptr<AlignedPtr> GeTensorImpl::GetAlignedPtr() {
  if (tensor_data_.impl_ != nullptr) {
    return tensor_data_.impl_->GetAlignedPtr();
  }
  return nullptr;
}

GeTensorImpl::GeTensorImpl(const GeTensorImpl &other) : GeTensorImpl() {
  *this = other;
}

GeTensorImpl &GeTensorImpl::operator=(const GeTensorImpl &other) {
  if (&other != this) {
    if (other.tensor_def_.GetProtoOwner() != nullptr) {
      // Old scene, share tensor_def, tensor_desc, tensor_data with `other`
      tensor_def_ = other.tensor_def_;
      // 这里修改了
      desc_ = other.desc_;
      if (tensor_data_.impl_ != nullptr && desc_.impl_ != nullptr) {
        tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
      }
      BuildAlignerPtrWithProtoData();
    } else {
      // share tensor_data, do not share tensor_desc, tensor_def is null
      desc_ = other.desc_;
      tensor_data_ = other.tensor_data_;
      if (tensor_data_.impl_ != nullptr && desc_.impl_ != nullptr) {
        tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
      }
    }
  }
  return *this;
}

GeTensor::GeTensor() : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl())) {}

GeTensor::GeTensor(GeTensor &&other) noexcept : impl_(std::move(other.impl_)) {}

GeTensor::GeTensor(GeTensorImplPtr impl) : impl_(std::move(impl)) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(tensor_desc))) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const vector<uint8_t> &data)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(tensor_desc, data))) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const uint8_t *data, size_t size)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(tensor_desc, data, size))) {}

GeTensor::GeTensor(GeTensorDesc &&tensor_desc, vector<uint8_t> &&data)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(std::move(tensor_desc), std::move(data)))) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const Buffer &data)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(tensor_desc, data))) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr, size_t size)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(tensor_desc, aligned_ptr, size))) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, size_t size)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(tensor_desc, size))) {}

GeTensor::GeTensor(const ProtoMsgOwner &proto_owner, proto::TensorDef *protoMsg)
    : impl_(std::shared_ptr<GeTensorImpl>(new GeTensorImpl(proto_owner, protoMsg))) {}

GeTensor::~GeTensor() = default;

void GeTensor::BuildAlignerPtrWithProtoData() {
  impl_->BuildAlignerPtrWithProtoData();
}

const GeTensorDesc &GeTensor::GetTensorDesc() const { return DescReference(); }

GeTensorDesc &GeTensor::MutableTensorDesc() { return DescReference(); }

GeTensorDesc &GeTensor::DescReference() const {
  return impl_->DescReference();
}

void GeTensor::SetTensorDesc(const GeTensorDesc &tensor_desc) { DescReference() = tensor_desc; }

graphStatus GeTensor::SetData(vector<uint8_t> &&data) {
  return impl_->SetData(data);
}

graphStatus GeTensor::SetData(const vector<uint8_t> &data) {
  return impl_->SetData(data);
}

graphStatus GeTensor::SetData(const uint8_t *data, size_t size) {
  return impl_->SetData(data, size);
}

graphStatus GeTensor::SetData(const Buffer &data) {
  return impl_->SetData(data);
}

graphStatus GeTensor::SetData(const TensorData &data) {
  return SetData(data.data(), data.size());
}

graphStatus GeTensor::SetData(uint8_t *data, size_t size, const AlignedPtr::Deleter &delete_fuc) {
  return impl_->SetData(data, size, delete_fuc);
}

void GeTensor::ClearData() {
  impl_->ClearData();
}

GeTensor GeTensor::Clone() const {
  GeTensor tensor;
  impl_->Clone(*(tensor.impl_));
  return tensor;
}

GeTensor::GeTensor(const GeTensor &other)
    : impl_((new GeTensorImpl(*(other.impl_)))) {}

GeTensor &GeTensor::operator=(const GeTensor &other) {
  if (&other != this) {
    *impl_ = *(other.impl_);
  }
  return *this;
}

std::shared_ptr<AlignedPtr> GeTensor::GetAlignedPtr() {
  return impl_->GetAlignedPtr();
}

const TensorData &GeTensor::GetData() const {
  return impl_->GetData();
}
TensorData &GeTensor::MutableData() {
  return impl_->MutableData();
}
// zero copy SetData
void GeTensor::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, size_t size) {
  impl_->SetData(std::move(aligned_ptr), size);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetSize(const GeTensorDesc &tensor_desc,
                                                                                int64_t &size) {
  if (tensor_desc.impl_ != nullptr) {
    // 所有的impl_->tensor_descriptor_.GetProtoMsg()都要替换为any map或者直接调用成员方法
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    GE_CHECK_NOTNULL(tensor_descriptor_msg);
    size = static_cast<int64_t>(tensor_descriptor_msg->size());
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetSize(GeTensorDesc &tensor_desc, int64_t size) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_size(size);
    }
  } else {
    GELOGW("Tensor utils set size failed, tensor desc impl is nullptr.");
  }
}

uint32_t TensorUtils::GetWeightSize(const GeTensorDesc &tensor_desc) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      return static_cast<uint32_t>(tensor_descriptor_msg->weight_size());
    }
  }
  return 0;
}

uint32_t TensorUtils::GetWeightSize(const GeTensor &tensor) { return GetWeightSize(tensor.GetTensorDesc()); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t TensorUtils::GetWeightSize(const ConstGeTensorPtr &tensor_ptr) {
  if (tensor_ptr == nullptr) {
    return 0;
  }
  return GetWeightSize(*tensor_ptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint8_t *TensorUtils::GetWeightAddr(const ConstGeTensorPtr &tensor_ptr,
                                                                                   uint8_t *base) {
  if (tensor_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "param tensor_ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] tensor_ptr is null.");
    return nullptr;
  }
  return GetWeightAddr(*tensor_ptr, base);
}

uint8_t *TensorUtils::GetWeightAddr(const GeTensor &tensor, uint8_t *base) {
  if (base == nullptr) {
    REPORT_INNER_ERROR("E19999", "param base is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] base is null.");
    return nullptr;
  }
  int64_t weight_data_offset = 0;
  if (GetDataOffset(tensor.GetTensorDesc(), weight_data_offset) != GRAPH_SUCCESS) return nullptr;

  if (weight_data_offset == 0) {
    // The weight of offset 0 is still in const op, still get from ATTR_NAME_WEIGHTS.
    return const_cast<uint8_t *>(tensor.GetData().data());
  }

  return base + weight_data_offset;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetWeightSize(GeTensorDesc &tensor_desc,
                                                                               uint32_t size) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_weight_size(size);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetReuseInput(const GeTensorDesc &tensor_desc,
                                                                                      bool &flag) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    GE_CHECK_NOTNULL(tensor_descriptor_msg);
    flag = tensor_descriptor_msg->reuse_input();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInput(GeTensorDesc &tensor_desc, bool flag) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_reuse_input(flag);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetOutputTensor(const GeTensorDesc &tensor_desc,
                                                                                        bool &flag) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    GE_CHECK_NOTNULL(tensor_descriptor_msg);
    flag = tensor_descriptor_msg->output_tensor();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetOutputTensor(GeTensorDesc &tensor_desc, bool flag) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_output_tensor(flag);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDeviceType(const GeTensorDesc &tensor_desc,
                                                                                      DeviceType &type) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    GE_CHECK_NOTNULL(tensor_descriptor_msg);
    string type_str = tensor_descriptor_msg->device_type();
    auto iter = kStrToDeviceMap.find(type_str);
    if (iter != kStrToDeviceMap.end()) {
      type = iter->second;
    } else {
      REPORT_CALL_ERROR("E19999", "GetDeviceType failed, device_type=%s.", type_str.c_str());
      GELOGE(GRAPH_FAILED, "[Get][DeviceType] failed, data_type=%s.", type_str.c_str());
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDeviceType(GeTensorDesc &tensor_desc,
                                                                               DeviceType type) {
  auto iter = kDeviceToStrMap.find(type);
  std::string type_str;
  if (iter != kDeviceToStrMap.end()) {
    type_str = iter->second;
  } else {
    GELOGW("[Set][DeviceType] not found device type[%d].", static_cast<int32_t>(type));
  }
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_device_type(type_str);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetInputTensor(const GeTensorDesc &tensor_desc,
                                                                                       bool &flag) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    GE_CHECK_NOTNULL(tensor_descriptor_msg);
    flag = tensor_descriptor_msg->input_tensor();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetInputTensor(GeTensorDesc &tensor_desc, bool flag) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_input_tensor(flag);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRealDimCnt(const GeTensorDesc &tensor_desc,
                                                                                      uint32_t &cnt) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    GE_CHECK_NOTNULL(tensor_descriptor_msg);
    cnt = static_cast<uint32_t>(tensor_descriptor_msg->real_dim_cnt());
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRealDimCnt(GeTensorDesc &tensor_desc,
                                                                               uint32_t cnt) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_real_dim_cnt(cnt);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetReuseInputIndex(const GeTensorDesc &tensor_desc, uint32_t &idx) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    GE_CHECK_NOTNULL(tensor_descriptor_msg);

    idx = static_cast<uint32_t>(tensor_descriptor_msg->reuse_input_index());
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInputIndex(GeTensorDesc &tensor_desc,
                                                                                    uint32_t idx) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_reuse_input_index(idx);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDataOffset(const GeTensorDesc &tensor_desc,
                                                                                      int64_t &offset) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      offset = tensor_descriptor_msg->data_offset();
      return GRAPH_SUCCESS;
    } else {
      GELOGW("tensor_descriptor_msg is nullptr.");
      return GRAPH_FAILED;
    }
  } else {
    GELOGW("[Get][DataOffset] tensor desc impl is nullptr.");
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDataOffset(GeTensorDesc &tensor_desc,
                                                                               int64_t offset) {
  if (tensor_desc.impl_ != nullptr) {
    auto tensor_descriptor_msg = tensor_desc.impl_->tensor_descriptor_.GetProtoMsg();
    if (tensor_descriptor_msg != nullptr) {
      tensor_descriptor_msg->set_data_offset(offset);
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRC(const GeTensorDesc &tensor_desc,
                                                                              uint32_t &rc) {
  return AttrUtils::GetInt(&tensor_desc, TENSOR_UTILS_RC, rc) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRC(GeTensorDesc &tensor_desc, uint32_t rc) {
  (void)AttrUtils::SetInt(&tensor_desc, TENSOR_UTILS_RC, rc);
}

GeTensor TensorUtils::CreateShareTensor(const GeTensor &other) {
  GeTensor tensor;
  ShareTensor(other, tensor);
  return tensor;
}

GeTensor TensorUtils::CreateShareTensor(const GeTensorDesc &tensorDesc,
                                        std::shared_ptr<AlignedPtr> aligned_ptr,
                                        size_t size) {
  GeTensor tensor(tensorDesc);
  if (tensor.impl_ != nullptr) {
    ShareAlignedPtr(std::move(aligned_ptr), size, tensor.impl_->tensor_data_);
  }
  return tensor;
}

void TensorUtils::ShareTensor(const GeTensor &from, GeTensor &to) {
  if (&from == &to) {
    return;
  }
  if (from.impl_ != nullptr && to.impl_ != nullptr) {
    if (from.impl_->tensor_def_.GetProtoOwner() != nullptr) {
      // 这种场景下看原来的逻辑，已经没有什么是不是共享的了，所以直接改成了impl共享，幸好impl是shared ptr
      // 但是之前似乎有个啥逻辑。是假定可以把shared ptr当成unique用的，得风暴下，记不得了
      to.impl_ = from.impl_;
    } else {
      // share tensor_data, do not share tensor_desc, tensor_def is null
      to.impl_->desc_ = from.impl_->desc_;
      to.impl_->tensor_data_ = from.impl_->tensor_data_;
      to.impl_->tensor_data_.impl_->tensor_descriptor_ = to.impl_->desc_.impl_;
    }
  }
}
void TensorUtils::ShareTensorData(const TensorData &from, TensorData &to) {
  if (&from == &to) {
    return;
  }
  // Share data
  if (from.impl_ != nullptr && to.impl_ != nullptr) {
    to.impl_->tensor_descriptor_ = from.impl_->tensor_descriptor_;
    to.impl_->aligned_ptr_ = from.impl_->aligned_ptr_;
    to.impl_->length_ = from.impl_->length_;
  }
}
TensorData TensorUtils::CreateShareTensorData(const TensorData &other) {
  TensorData td;
  ShareTensorData(other, td);
  return td;
}
void TensorUtils::ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, size_t size, TensorData &to) {
  if (to.impl_ != nullptr) {
    to.impl_->aligned_ptr_ = std::move(ptr);
    to.impl_->length_ = size;
  }
}
void TensorUtils::ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, size_t size, GeTensor &to) {
  if (to.impl_ != nullptr) {
    ShareAlignedPtr(std::move(ptr), size, to.impl_->tensor_data_);
  }
}
// UT
void TensorUtils::CopyTensor(const GeTensor &from, GeTensor &to) {
  if (&from == &to) {
    return;
  }
  if (from.impl_ == nullptr || to.impl_ == nullptr) {
    return;
  }
  if (from.impl_->tensor_def_.GetProtoOwner() != nullptr) {
    to.impl_->tensor_def_.CopyValueFrom(from.impl_->tensor_def_);
    to.impl_->desc_.impl_ = GeTensorDesc(to.impl_->tensor_def_.GetProtoOwner(),
                                         to.impl_->tensor_def_.GetProtoMsg()->mutable_desc()).impl_;
    to.impl_->desc_.impl_->attrs_ = from.impl_->desc_.impl_->attrs_;
    to.impl_->tensor_data_.impl_->tensor_descriptor_ = to.impl_->desc_.impl_;
    to.BuildAlignerPtrWithProtoData();
  } else {
    // tensor_def is null, copy tensor_data, tensor_desc
    to.impl_->desc_ = from.impl_->desc_;
    to.impl_->tensor_data_.SetData(from.impl_->tensor_data_);
    to.impl_->tensor_data_.impl_->tensor_descriptor_ = to.impl_->desc_.impl_;
  }
}
}  // namespace ge
