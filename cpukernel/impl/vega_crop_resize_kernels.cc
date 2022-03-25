
/*
 * Copyright (c) Deepglint Co., Ltd. 2022. All rights reserved.
 * by zhenxiongchen@deepglint.com
 */
#include "vega_crop_resize_kernels.h"
#include "vega_type.h"
namespace  {
const char *VEGA_CROP_RESIZE = "VegaCropResize";
}
namespace aicpu  {
    typedef struct {
                VegaMatrix in;
                VegaMatrix out;
                float mean[3];
                float scale;
    }VegaCropResizeParam;

uint32_t VegaCropResizeCpuKernel::Compute(CpuKernelContext &ctx)
{
    Tensor *param_tensor = ctx.Input(0);
    if (param_tensor == nullptr) {
        return 1;
    }

    VegaCropResizeParam *param = (VegaCropResizeParam *)param_tensor->GetData();

    if(param->out.type != (unsigned int )BGRPacked){
        return 1;
    }

    unsigned char *img_in = (unsigned char *) param->in.addr;
    if(param->in.type == (unsigned int )NV12){
        bool to_resize=true;
        bool to_copy=false;
 
        int img_in_w=param->in.w;
        if(img_in_w %2 !=0 ){
            img_in_w--;
        }
        cv::Rect r=cv::Rect(param->in.roi_x,param->in.roi_y,param->in.roi_w,param->in.roi_h);
        auto tl = r.tl();
        auto br = r.br();
        if(tl.x % 2 != 0) tl.x++;
        if(tl.y % 2 != 0) tl.y++;
        if(br.x % 2 != 0) br.x--;
        if(br.y % 2 != 0) br.y--;
        cv::Rect newRoi(tl, br);

        cv::Mat bgr;
        if(param->in.w==param->out.w&&param->in.h==param->out.h&&
            param->in.roi_x==param->out.roi_x&&param->in.roi_y==param->out.roi_y&&
            param->in.roi_w==param->out.roi_w&&param->in.roi_h==param->out.roi_h&&
            param->out.w==param->out.roi_w&&param->out.roi_x==0&&param->out.roi_y==0){
            bgr=cv::Mat(param->out.h,param->out.w,CV_8UC3,(uchar*)param->out.addr,param->out.s_w);
            to_resize=false;
        }else if(param->in.roi_w==param->out.roi_w&&param->in.roi_h==param->out.roi_h){
            to_resize=false;
            to_copy=true;
        }
        if(to_resize||to_copy){
            bgr=cv::Mat(param->in.h/2*2,img_in_w,CV_8UC3);
        }
        bgr=cv::Mat(param->in.h/2*2,img_in_w,CV_8UC3);
        cv::Size sz(img_in_w,param->in.h/2*3);
        cv::Mat nv12=cv::Mat(sz, CV_8UC1, (uchar*)param->in.addr,param->in.s_w);
        if(param->out.w*param->out.h > param->out.roi_w*param->out.roi_h){
            memset((uchar*)param->out.addr,0,param->out.s_w*param->out.s_h);
        }
        cv::cvtColor(nv12, bgr,cv::COLOR_YUV2BGR_NV12);
        if(to_copy) {
            auto dst=cv::Mat(param->out.h,param->out.w,CV_8UC3,(uchar*)param->out.addr,param->out.s_w);
            auto dst_roi=dst(newRoi);
            cv::Mat img_in_roi= bgr(newRoi);
            img_in_roi.copyTo(dst_roi);
        }
        if(to_resize){
            cv::Mat img_in_roi= bgr(newRoi);
            cv::Mat img_out = cv::Mat(param->out.h, param->out.w,CV_8UC3,(uchar*)param->out.addr,param->out.s_w);
            cv::Mat img_out_roi= img_out(cv::Rect(param->out.roi_x,param->out.roi_y,param->out.roi_w,param->out.roi_h));
            cv::resize(img_in_roi, img_out_roi, img_out_roi.size(),0,0,cv::INTER_LINEAR);
        }
        return 0;
    }

    if(param->in.type == (unsigned int )BGRPacked && param->out.type == (unsigned int )BGRPacked){
        cv::Mat img_in = cv::Mat(param->in.h, param->in.w,CV_8UC3,(uchar*)param->in.addr,param->in.s_w);
        cv::Mat img_in_roi= img_in(cv::Rect(param->in.roi_x,param->in.roi_y,param->in.roi_w,param->in.roi_h));
        cv::Mat img_out = cv::Mat(param->out.h, param->out.w,CV_8UC3,(uchar*)param->out.addr,param->out.s_w);
        cv::Mat img_out_roi= img_out(cv::Rect(param->out.roi_x,param->out.roi_y,param->out.roi_w,param->out.roi_h));
        cv::resize(img_in_roi, img_out_roi, img_out_roi.size(),0,0,cv::INTER_LINEAR);
        return 0;
    }
 
    return 1;
}

REGISTER_CPU_KERNEL(VEGA_CROP_RESIZE, VegaCropResizeCpuKernel);
} // namespace aicpu
