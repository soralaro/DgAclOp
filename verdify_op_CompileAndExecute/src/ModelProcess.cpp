/*
 * Copyright(C) 2022. Deepglint Technologies Co.,Ltd. All rights reserved.
 * by chenzhenxiong
 */

#include "ModelProcess.h"

ModelProcess::ModelProcess()
{

}

ModelProcess::~ModelProcess()
{
}



int ModelProcess::ModelInference(std::vector<void *> &inputBufs, std::vector<size_t> &inputSizes,
    std::vector<void *> &ouputBufs, std::vector<size_t> &outputSizes, aclrtStream stream)
{
    cout << "ModelProcess:Begin to inference." << endl;
    aclmdlDataset *input = nullptr;
    std::string op_type = "WarpAffine";

    aclTensorDesc *img_in_desc = aclCreateTensorDesc(
            ACL_INT8, img_in_shape_.size(), img_in_shape_.data(), ACL_FORMAT_NHWC);
    aclTensorDesc *Trans_M_desc = aclCreateTensorDesc(
            ACL_FLOAT, Trans_M_shape_.size(), Trans_M_shape_.data(), ACL_FORMAT_ND);
    aclTensorDesc *img_out_desc = aclCreateTensorDesc(
            ACL_INT8, img_out_shape_.size(), img_out_shape_.data(), ACL_FORMAT_NHWC);
    std::vector<aclTensorDesc*> input_desc = {img_in_desc, Trans_M_desc};
    std::vector<aclTensorDesc*> output_desc = {img_out_desc};

    auto src_size = aclGetTensorDescSize(img_in_desc);
    aclDataBuffer *img_in_buffer = aclCreateDataBuffer(inputBufs[0], src_size);
    src_size = aclGetTensorDescSize(Trans_M_desc);
    aclDataBuffer *Trans_M_buffer = aclCreateDataBuffer(inputBufs[1], src_size);
    src_size = aclGetTensorDescSize(img_out_desc);
    aclDataBuffer *img_out_buffer = aclCreateDataBuffer(ouputBufs[0], src_size);

    std::vector<aclDataBuffer*> input_buffer = {img_in_buffer, Trans_M_buffer};
    std::vector<aclDataBuffer*> output_buffer = {img_out_buffer};
    // Set OP attribute
    aclopAttr *op_attr = aclopCreateAttr();
    std::vector<int64_t> img_in_size={0,0,img_in_shape_[2],img_in_shape_[1]};//x,y,w,h
    auto ret=aclopSetAttrListInt(op_attr, "img_in_size", 4, img_in_size.data());
    if (ret != ACL_SUCCESS) {
        cout<<"Failed to aclopSetAttrListInt img_in_size "<<endl;
    }
    std::vector<int64_t> img_out_size={0,0,img_out_shape_[2],img_out_shape_[1]};//x,y,w,h

    ret=aclopSetAttrListInt(op_attr, "img_out_size", 4, img_out_size.data());
    if (ret != ACL_SUCCESS) {
        cout<<"Failed to aclopSetAttrListInt img_out_size "<<endl;
    }

    std::cout << op_type << " aclopExecuteV2 start " << op_compile <<" dst size "<<img_out_size[2]<< std::endl;
    ret = aclopExecuteV2(
            op_type.c_str(),
            inputBufs.size(),
            input_desc.data(),
            input_buffer.data(),
            ouputBufs.size(),
            output_desc.data(),
            output_buffer.data(),
            op_attr, stream);
    if (ret != ACL_SUCCESS) {
        cout << "Failed to aclopExecuteV2 " << endl;
        cout << op_type << " aclopCompileAndExecute start" << endl;
        ret = aclopCompileAndExecute(
                op_type.c_str(),
                inputBufs.size(), input_desc.data(), input_buffer.data(),
                ouputBufs.size(), output_desc.data(), output_buffer.data(),
                op_attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream);
        if (ret != ACL_SUCCESS) {
            cout << "Failed to aclopCompileAndExecute " << endl;
        }
    }
    aclrtSynchronizeStream(stream);
    cout << op_type << " aclop end" << endl;
    aclDestroyDataBuffer(img_in_buffer);
    aclDestroyDataBuffer(Trans_M_buffer);
    aclDestroyDataBuffer(img_out_buffer);

    aclDestroyTensorDesc(img_in_desc);
    aclDestroyTensorDesc(Trans_M_desc);
    aclDestroyTensorDesc(img_out_desc);
    return ACL_ERROR_NONE;
}


