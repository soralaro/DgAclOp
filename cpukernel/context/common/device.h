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
#ifndef AICPU_CONTEXT_COMMON_DEVICE_H_
#define AICPU_CONTEXT_COMMON_DEVICE_H_

#include "sharder.h"

namespace aicpu {
class Device {
 public:
  explicit Device(DeviceType device);

  ~Device();

  /*
   * get device type.
   * @return DeviceType: HOST/DEVICE
   */
  DeviceType GetDeviceType() const;

  /*
   * get sharder.
   * @return Sharder *: host or device sharder
   */
  const Sharder *GetSharder() const;

 private:
  Device(const Device &) = delete;
  Device(Device &&) = delete;
  Device &operator=(const Device &) = delete;
  Device &operator=(Device &&) = delete;

  /*
   * init sharder.
   * param device: type of device
   * @return Sharder *: not null->success, null->success
   */
  Sharder *InitSharder(DeviceType device);

 private:
  DeviceType device_;  // type of device
  Sharder *sharder_;
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_DEVICE_H_
