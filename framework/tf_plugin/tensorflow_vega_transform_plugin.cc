/**
 * Copyright 2022 deepglint Technologies Co., Ltd
 * by zhenxiongchen@deepglint.com
 */

/*!
 * \file cast_plugin.cpp
 * \brief
 */
#include "register/register.h"

#include "op_log.h"

namespace domi {

Status AutoMappingFnVegaTransform(const google::protobuf::Message* op_src, ge::Operator& op) {
  return SUCCESS;
}

REGISTER_CUSTOM_OP("VegaTransform")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("VegaTransform")
    .ParseParamsFn(AutoMappingFnVegaTransform)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
