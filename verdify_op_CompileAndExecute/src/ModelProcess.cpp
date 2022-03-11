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
    std::string op_type = "FaceAlign";

    aclTensorDesc *image_desc = aclCreateTensorDesc(
            ACL_INT8, image_shape_.size(), image_shape_.data(), ACL_FORMAT_NHWC);
    aclTensorDesc *keypoints_desc = aclCreateTensorDesc(
            ACL_FLOAT, keypoints_shape_.size(), keypoints_shape_.data(), ACL_FORMAT_ND);
    aclTensorDesc *face_num_desc = aclCreateTensorDesc(
            ACL_INT32, face_num_shape_.size(), face_num_shape_.data(), ACL_FORMAT_ND);
    aclTensorDesc *aligned_image_desc = aclCreateTensorDesc(
            ACL_INT8, aligned_image_shape_.size(), aligned_image_shape_.data(), ACL_FORMAT_ND);
    std::vector<aclTensorDesc*> input_desc = {image_desc, keypoints_desc,face_num_desc};
    std::vector<aclTensorDesc*> output_desc = {aligned_image_desc};

    auto src_size = aclGetTensorDescSize(image_desc);
    aclDataBuffer *image_buffer = aclCreateDataBuffer(inputBufs[0], src_size);
    src_size = aclGetTensorDescSize(keypoints_desc);
    aclDataBuffer *keypoints_buffer = aclCreateDataBuffer(inputBufs[1], src_size);
    src_size = aclGetTensorDescSize(face_num_desc);
    aclDataBuffer *face_num_buffer = aclCreateDataBuffer(inputBufs[2], src_size);
    src_size = aclGetTensorDescSize(aligned_image_desc);
    aclDataBuffer *aligned_image_buffer = aclCreateDataBuffer(ouputBufs[0], src_size);

    std::vector<aclDataBuffer*> input_buffer = {image_buffer, keypoints_buffer,face_num_buffer};
    std::vector<aclDataBuffer*> output_buffer = {aligned_image_buffer};
    // Set OP attribute
    aclopAttr *op_attr = aclopCreateAttr();
    std::vector<int64_t> face_size={112,112};
    auto ret=aclopSetAttrListInt(op_attr, "face_size", 2, face_size.data());
    if (ret != ACL_SUCCESS) {
        cout<<"Failed to aclopSetAttrListInt face_size "<<endl;
    }
    std::vector<int64_t> default_keypoint={40,45,72,45,52,65,42,82,72,82};
    ret=aclopSetAttrListInt(op_attr, "default_keypoint", default_keypoint.size(), default_keypoint.data());
    if (ret != ACL_SUCCESS) {
        cout<<"Failed to aclopSetAttrListInt default_keypoint "<<endl;
    }

    cout << "aclopCompileAndExecute start" << endl;
    ret = aclopCompileAndExecute(
            op_type.c_str(),
            inputBufs.size(), input_desc.data(), input_buffer.data(),
            ouputBufs.size(), output_desc.data(), output_buffer.data(),
            op_attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream);
    if (ret != ACL_SUCCESS) {
        cout<<"Failed to aclopCompileAndExecute "<<endl;
    }
    aclrtSynchronizeStream(stream);
    cout << "aclopCompileAndExecute end" << endl;
    aclDestroyDataBuffer(image_buffer);
    aclDestroyDataBuffer(keypoints_buffer);
    aclDestroyDataBuffer(face_num_buffer);
    aclDestroyDataBuffer(aligned_image_buffer);

    aclDestroyTensorDesc(image_desc);
    aclDestroyTensorDesc(keypoints_desc);
    aclDestroyTensorDesc(face_num_desc);
    aclDestroyTensorDesc(aligned_image_desc);
    return ACL_ERROR_NONE;
}


