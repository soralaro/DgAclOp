
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of FaceAlign
 */
#include "face_align_kernels.h"
#define FACE_KEYPOINT_NUM 5
namespace  {
const char *FACE_ALIGN = "FaceAlign";
}

namespace aicpu  {
uint32_t FaceAlignCpuKernel::Compute(CpuKernelContext &ctx)
{
    //get input tensor
    Tensor *image_tensor = ctx.Input(0);
    if (image_tensor == nullptr) {
        return 1;
    }

    Tensor *keypoint_tensor = ctx.Input(1);
    if (keypoint_tensor == nullptr) {
        return 1;
    }

    Tensor *face_num_tensor = ctx.Input(2);
    if (keypoint_tensor == nullptr) {
        return 1;
    }
    //get attr
    AttrValue* face_size_attr = ctx.GetAttr("face_size");
    if ( face_size_attr == nullptr){
        return 1;
    }
    std::vector<int64_t> face_size = face_size_attr->GetListInt();
    if ( face_size.size() != 2){
        return 1;
    }
    AttrValue* default_keypoint_attr = ctx.GetAttr("default_keypoint");
    if ( default_keypoint_attr == nullptr){
        return 1;
    }
    std::vector<int64_t> default_keypoint = default_keypoint_attr->GetListInt();
    if ( default_keypoint.size() != 10){
        return 1;
    }
    //get output ptr
    uint8_t* output_ptr = (uint8_t*)ctx.Output(0)->GetData();
    if (output_ptr == nullptr) {
        return 1;
    }
    //get image shape
    std::shared_ptr<TensorShape> image_tensor_shape = image_tensor->GetTensorShape();
    std::vector<int64_t> image_shapes = image_tensor_shape->GetDimSizes(); //NHWC
    if(image_shapes.size() < 4){
        return -1;
    }

    //get image data
    Mat img = Mat(image_shapes[1], image_shapes[2], CV_8UC3);
    // Mat img = Mat(480, 480, CV_8UC3);
    img.data = (uchar*)image_tensor->GetData();
    //get keypoint data
    float* keypoint_data_ptr = (float*)keypoint_tensor->GetData();
    //get face number data
    int32_t face_num = *(int32_t*)face_num_tensor->GetData();

    std::vector<Point2f> src_face_keypoints(FACE_KEYPOINT_NUM);
    std::vector<Point2f> dst_face_keypoints(FACE_KEYPOINT_NUM);
    //warp default keypoint
    for(int i = 0; i < FACE_KEYPOINT_NUM; i++){
        src_face_keypoints[i].x = (float)default_keypoint[i * 2];
        src_face_keypoints[i].y = (float)default_keypoint[i * 2 + 1];
    }

    for(int i = 0; i < face_num; i++){
        //warp target keypoint
        dst_face_keypoints[0].x = keypoint_data_ptr[i * 10 + 0];
        dst_face_keypoints[0].y = keypoint_data_ptr[i * 10 + 1];
        dst_face_keypoints[1].x = keypoint_data_ptr[i * 10 + 2];
        dst_face_keypoints[1].y = keypoint_data_ptr[i * 10 + 3];
        dst_face_keypoints[2].x = keypoint_data_ptr[i * 10 + 4];
        dst_face_keypoints[2].y = keypoint_data_ptr[i * 10 + 5];
        dst_face_keypoints[3].x = keypoint_data_ptr[i * 10 + 6];
        dst_face_keypoints[3].y = keypoint_data_ptr[i * 10 + 7];
        dst_face_keypoints[4].x = keypoint_data_ptr[i * 10 + 8];
        dst_face_keypoints[4].y = keypoint_data_ptr[i * 10 + 9];

        Mat M = estimateAffine2D(dst_face_keypoints, src_face_keypoints);
        Mat face_alinged;
        warpAffine(img, face_alinged, M, Size(face_size[0],face_size[1]));
        int image_size = face_alinged.cols * face_alinged.rows * face_alinged.channels();
        memcpy(output_ptr + i * image_size, face_alinged.data, image_size);
    }
    return 0;
}

REGISTER_CPU_KERNEL(FACE_ALIGN, FaceAlignCpuKernel);
} // namespace aicpu
