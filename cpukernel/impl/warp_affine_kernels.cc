
/*
 * Copyright (c) Deepglint Co., Ltd. 2022. All rights reserved.
 * by zhenxiongchen@deepglint.com
 */
#include "warp_affine_kernels.h"
namespace  {
const char *WARP_AFFINE = "WarpAffine";
}
namespace aicpu  {
uint32_t WarpAffineCpuKernel::Compute(CpuKernelContext &ctx)
{

    Tensor *img_in_tensor = ctx.Input(0);
    if (img_in_tensor == nullptr) {
        return 1;
    }

    Tensor *trans_M_tensor = ctx.Input(1);
    if (trans_M_tensor == nullptr) {
        return 1;
    }

    Tensor *img_out_tensor = ctx.Output(0);
    if (img_out_tensor == nullptr) {
        return 1;
    }

    //get attr
    AttrValue* img_in_size_attr = ctx.GetAttr("img_in_size");
    if ( img_in_size_attr == nullptr){
        return 1;
    }
    std::vector<int64_t> img_in_size = img_in_size_attr->GetListInt();
    if ( img_in_size.size() != 4){
        return 1;
    }

    AttrValue* img_out_size_attr = ctx.GetAttr("img_out_size");
    if ( img_out_size_attr == nullptr){
        return 1;
    }
    std::vector<int64_t> img_out_size = img_out_size_attr->GetListInt();
    if ( img_out_size.size() != 4){
        return 1;
    }

    std::shared_ptr<TensorShape> img_in_tensor_shape = img_in_tensor->GetTensorShape();
    std::vector<int64_t> img_in_shapes = img_in_tensor_shape->GetDimSizes(); //NHWC
    if(img_in_shapes.size() < 4){
        return -1;
    }

    std::shared_ptr<TensorShape> trans_M_tensor_shape = trans_M_tensor->GetTensorShape();
    std::vector<int64_t> trans_M_shapes = trans_M_tensor_shape->GetDimSizes();
    if(trans_M_shapes.size() < 2){
        return -1;
    }

    std::shared_ptr<TensorShape> img_out_tensor_shape = img_out_tensor->GetTensorShape();
    std::vector<int64_t> img_out_shapes = img_out_tensor_shape->GetDimSizes(); //NHWC
    if(img_out_shapes.size() < 4){
        return -1;
    }

    cv::Mat img_in = cv::Mat(img_in_shapes[1], img_in_shapes[2],CV_8UC3,(uchar*)img_in_tensor->GetData());
    cv::Mat img_in_roi= img_in(cv::Rect(img_in_size[0],img_in_size[1],img_in_size[2],img_in_size[3]));
    cv::Mat Trans_M=cv::Mat(trans_M_shapes[0],trans_M_shapes[1],CV_32FC1,(float *)trans_M_tensor->GetData());

    cv::Mat img_out = cv::Mat(img_out_shapes[1], img_out_shapes[2],CV_8UC3,(uchar*)img_out_tensor->GetData());
    cv::Mat img_out_roi= img_out(cv::Rect(img_out_size[0],img_out_size[1],img_out_size[2],img_out_size[3]));
    if(img_in_roi.empty()||img_out_roi.empty()){
       return -1;
    }
    cv::warpAffine(img_in_roi,
                   img_out_roi,
                   Trans_M,
                   cv::Size(img_out_size[2],img_out_size[3]),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0,0,0));
    return 0;
}

REGISTER_CPU_KERNEL(WARP_AFFINE, WarpAffineCpuKernel);
} // namespace aicpu
