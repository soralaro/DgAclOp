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
#include "device.h"

#include <new>

#include "device_sharder.h"
#include "host_sharder.h"

namespace aicpu {
Device::Device(DeviceType device) {
  device_ = device;
  sharder_ = InitSharder(device);
}

Device::~Device() {
  if (sharder_ != nullptr) {
    delete sharder_;
  }
}

/*
 * get device type.
 * @return DeviceType: HOST/DEVICE
 */
DeviceType Device::GetDeviceType() const { return device_; }

/*
 * get sharder.
 * @return Sharder *: host or device sharder
 */
const Sharder *Device::GetSharder() const {
  if (sharder_ != nullptr) {
    return sharder_;
  }
  return nullptr;
}

/*
 * init sharder.
 * param device: type of device
 * @return Sharder *: not null->success, null->success
 */
Sharder *Device::InitSharder(DeviceType device_) {
  if (device_ == DEVICE) {
    return new (std::nothrow) DeviceSharder(device_);
  } else {
    return new (std::nothrow) HostSharder(device_);
  }
}
}  // namespace aicpu
