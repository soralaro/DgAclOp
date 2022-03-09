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
#ifndef AICPU_CONTEXT_COMMON_LOG_H_
#define AICPU_CONTEXT_COMMON_LOG_H_

#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "toolchain/slog.h"

#define GET_TID() syscall(__NR_gettid)
const char KERNEL_MODULE[] = "AICPU";

#ifdef RUN_TEST
#define KERNEL_LOG_DEBUG(fmt, ...)                                    \
  printf("[DEBUG] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, \
         __FILE__, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_INFO(fmt, ...)                                              \
  printf("[INFO] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_WARN(fmt, ...)                                              \
  printf("[WARN] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_ERROR(fmt, ...)                                    \
  printf("[ERROR] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, \
         __FILE__, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_EVENT(fmt, ...)                                    \
  printf("[EVENT] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, \
         __FILE__, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#else
#define KERNEL_LOG_DEBUG(fmt, ...)                                            \
  dlog_debug(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, \
             __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_INFO(fmt, ...)                                            \
  dlog_info(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, \
            __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_WARN(fmt, ...)                                            \
  dlog_warn(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, \
            __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_ERROR(fmt, ...)                                            \
  dlog_error(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, \
             __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_EVENT(fmt, ...)                                            \
  dlog_event(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, \
             __LINE__, GET_TID(), ##__VA_ARGS__)
#endif

#define KERNEL_CHECK_NULLPTR_VOID(value, logText...) \
  if (value == nullptr) {                            \
    KERNEL_LOG_ERROR(logText);                       \
    return;                                          \
  }

#define KERNEL_CHECK_FALSE(condition, errorCode, logText...)  \
  if (!(condition)) {                                         \
    KERNEL_LOG_ERROR(logText);                                \
    return errorCode;                                         \
  }

#define KERNEL_CHECK_NULLPTR(value, errorCode, logText...) \
  if (value == nullptr) {                                  \
    KERNEL_LOG_ERROR(logText);                             \
    return errorCode;                                      \
  }

#define KERNEL_CHECK_ASSIGN_64S_MULTI(A, B, result, errorCode)            \
  if ((A) != 0 && (B) != 0 && ((INT64_MAX) / (A)) <= (B)) {               \
    KERNEL_LOG_ERROR("Integer reversed multiA: %llu * multiB: %llu", (A), \
                     (B));                                                \
    return errorCode;                                                     \
  }                                                                       \
  (result) = ((A) * (B));

#endif  // AICPU_CONTEXT_COMMON_LOG_H_
