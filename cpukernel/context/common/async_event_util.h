/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef AICPU_CONTEXT_COMMON_ASYNC_EVENT_H_
#define AICPU_CONTEXT_COMMON_ASYNC_EVENT_H_

#include <functional>
#include "aicpu_context.h"

namespace aicpu {
typedef void (*NotifyWaitFunc)(void *notify_param, const uint32_t param_len);
typedef bool (*RegEventCbFunc)(const uint32_t event_id,
  const uint32_t sub_event_id, const std::function<void(void *)> &cb);

class AsyncEventUtil {
 public:
  static AsyncEventUtil &GetInstance();

  void NotifyWait(void *notify_param, const uint32_t param_len);

  bool RegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                  const std::function<void(void *)> &cb);
 private:
  AsyncEventUtil();
  ~AsyncEventUtil();
 private:
  void *sharder_;
  NotifyWaitFunc notify_wait_func_;
  RegEventCbFunc reg_event_cb_func_;
};
} // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_ASYNC_EVENT_H_