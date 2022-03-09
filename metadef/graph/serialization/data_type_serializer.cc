/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021.  All rights reserved.
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

#include "data_type_serializer.h"
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_log.h"
#include "graph/types.h"

namespace ge {
graphStatus DataTypeSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  ge::DataType value;
  graphStatus ret = av.GetValue(value);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get datatype attr.");
    return GRAPH_FAILED;
  }
  def.set_dt(static_cast<proto::DataType>(value));
  return GRAPH_SUCCESS;
}

graphStatus DataTypeSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(static_cast<DataType>(def.dt()));
}

REG_GEIR_SERIALIZER(DataTypeSerializer, GetTypeId<ge::DataType>(), proto::AttrDef::kDt);
}  // namespace ge
