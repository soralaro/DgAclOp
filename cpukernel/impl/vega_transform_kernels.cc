
/*
 * Copyright (c) Deepglint Co., Ltd. 2022. All rights reserved.
 * by zhenxiongchen@deepglint.com
 */
#include "vega_transform_kernels.h"
#include "vega_type.h"
#define   WARPA_FFINE 0
#define   WARPA_PERSPECTIVE 1
namespace  {
const char *VEGA_TRANSFORM = "VegaTransform";
}
namespace aicpu  {
        typedef struct {
                VegaMatrix in;
                VegaMatrix out;
                unsigned  int trans_m_size_w;
                unsigned  int trans_m_size_h;
                unsigned  int type;
                float trans_m[9];
    }VegaTransParam;
uint32_t VegaTransformCpuKernel::Compute(CpuKernelContext &ctx)
{
    Tensor *param_tensor = ctx.Input(0);
    if (param_tensor == nullptr) {
        return 1;
    }
    VegaTransParam *param = (VegaTransParam *)param_tensor->GetData();
    if((param->in.type != (unsigned int )BGRPacked && param->in.type != (unsigned int )NV12)|| param->out.type != (unsigned int )BGRPacked){
        return 1;
    }

    cv::Mat img_in_roi;
    if(param->in.type == (unsigned int )NV12){
        int img_in_w=param->in.w;
        if(img_in_w %2 !=0 ){
            img_in_w--;
        }
        auto bgr_in=cv::Mat(param->in.h/2*2,img_in_w,CV_8UC3);
        cv::Size sz(img_in_w,param->in.h/2*3);
        cv::Mat nv12=cv::Mat(sz, CV_8UC1, (uchar*)param->in.addr,param->in.s_w);
        cv::cvtColor(nv12, bgr_in,cv::COLOR_YUV2BGR_NV12);
        cv::Rect r=cv::Rect(param->in.roi_x,param->in.roi_y,param->in.roi_w,param->in.roi_h);
        auto tl = r.tl();
        auto br = r.br();
        if(tl.x % 2 != 0) tl.x++;
        if(tl.y % 2 != 0) tl.y++;
        if(br.x % 2 != 0) br.x--;
        if(br.y % 2 != 0) br.y--;
        cv::Rect newRoi(tl, br);
        img_in_roi= bgr_in(newRoi);
    }else{
        cv::Mat img_in=cv::Mat(param->in.h,param->in.w, CV_8UC3, (uchar*)param->in.addr,param->in.s_w);
        img_in_roi= img_in(cv::Rect(param->in.roi_x,param->in.roi_y,param->in.roi_w,param->in.roi_h));
    }

    cv::Mat Trans_M=cv::Mat(param->trans_m_size_h,param->trans_m_size_w,CV_32FC1,param->trans_m);

    cv::Mat img_out = cv::Mat(param->out.h, param->out.w,CV_8UC3,(uchar*)param->out.addr,param->out.s_w);
    cv::Mat img_out_roi= img_out(cv::Rect(param->out.roi_x,param->out.roi_y,param->out.roi_w,param->out.roi_h));
    if(img_in_roi.empty()||img_out_roi.empty()){
       return 1;
    }
    if(param->type==WARPA_FFINE){
        cv::warpAffine(img_in_roi,
            img_out_roi,
            Trans_M,
            cv::Size(param->out.roi_w,param->out.roi_h),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(0,0,0));
            return 0;
    }
    if(param->type==WARPA_PERSPECTIVE){
        cv::warpPerspective(img_in_roi, img_out_roi,Trans_M , cv::Size(param->out.roi_w,param->out.roi_h));
        return 0;
    }
    return 1;
}

REGISTER_CPU_KERNEL(VEGA_TRANSFORM, VegaTransformCpuKernel);
} // namespace aicpu
