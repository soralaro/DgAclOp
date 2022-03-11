#pragma once

#include "iostream"
#include "acl/acl.h"
#include "ModelProcess.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

struct ObjDetectInfo {
    float leftTopX;
    float leftTopY;
    float rightBotX;
    float rightBotY;
    float confidence;
    float classId;
};

class AclProcess{
public:
    AclProcess();
    ~AclProcess();
    int Init(int deviceId);
    int Process(Mat& img);
private:
    aclError PostProcess(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes, int face_num, int width, int height);

    std::vector<void *> inputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    aclrtContext context_;
    aclrtStream stream_;
    std::shared_ptr<ModelProcess> m_modelProcess;
};
