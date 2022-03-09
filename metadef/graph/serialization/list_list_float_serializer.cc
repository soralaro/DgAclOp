/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "list_list_float_serializer.h"
#include <vector>
#include "graph/debug/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_log.h"

namespace ge {
graphStatus ListListFloatSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  std::vector<std::vector<float>> list_list_value;
  graphStatus ret = av.GetValue(list_list_value);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_list_float attr.");
    return GRAPH_FAILED;
  }
  auto mutable_list_list = def.mutable_list_list_float();
  GE_CHECK_NOTNULL(mutable_list_list);
  mutable_list_list->clear_list_list_f();
  for (const auto &list_value : list_list_value) {
    auto list_f = mutable_list_list->add_list_list_f();
    GE_CHECK_NOTNULL(list_f);
    for (float val:list_value) {
      list_f->add_list_f(val);
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus ListListFloatSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  std::vector<std::vector<float>> values;
  for (int idx = 0; idx < def.list_list_float().list_list_f_size(); ++idx) {
    std::vector<float> vec;
    for (int i = 0; i < def.list_list_float().list_list_f(idx).list_f_size(); ++i) {
      vec.push_back(def.list_list_float().list_list_f(idx).list_f(i));
    }
    values.push_back(vec);
  }

  return av.SetValue(std::move(values));
}

REG_GEIR_SERIALIZER(ListListFloatSerializer,
                    GetTypeId<std::vector<std::vector<float>>>(), proto::AttrDef::kListListFloat);
}  // namespace ge