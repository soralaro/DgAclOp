#include "AclProcess.h"

AclProcess::AclProcess()
{

}
AclProcess::~AclProcess()
{
    for(int i=0;i<inputBuffers.size();i++){
        aclrtFree(inputBuffers[i]);
    }
    for(int i=0;i<outputBuffers.size();i++){
        aclrtFree(outputBuffers[i]);
    }
    m_modelProcess = nullptr;
    aclError ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "some tasks in stream not done, ret = " << ret <<endl;
    }
    cout << "all tasks in stream done" << endl;
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Destroy Stream faild, ret = " << ret <<endl;
    }
    cout << "Destroy Stream successfully" << endl;
    ret = aclrtDestroyContext(context_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Destroy Context faild, ret = " << ret <<endl;
    }
    cout << "Destroy Context successfully" << endl;
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to deinit acl, ret = " << ret <<endl;
    }
    cout << "acl deinit successfully" << endl;
}

int AclProcess::Init(int deviceId)
{
    //Init
    aclError ret = aclInit(nullptr); // Initialize ACL
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to init acl, ret = " << ret <<endl;
        return ret;
    }
    cout << "acl init successfully" << endl;
    ret = aclrtCreateContext(&context_, deviceId);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to set current context, ret = " << ret << endl;
        return ret;
    }
    cout << "Create context successfully" << endl;
    ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to set current context, ret = " << ret << endl;
        return ret;
    }
    cout << "set context successfully" << endl;
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to create stream, ret = " << ret << endl;
        return ret;
    }
    cout << "Create stream successfully" << endl;
    //Load model
    if (m_modelProcess == nullptr) {
        m_modelProcess = std::make_shared<ModelProcess>();
    }

    //get model input description and malloc them
    size_t inputSize = m_modelProcess->getNumInput();
    std::cout<<"inputSize "<<inputSize<<std::endl;
    for (size_t i = 0; i < inputSize; i++) {
        size_t bufferSize = m_modelProcess->getInputSizeByIndex(i);
        std::cout<<" i "<<i<<" bufferSize "<<bufferSize<<std::endl;
        void *inputBuffer = nullptr;
        //aclError ret = aclrtMalloc(&inputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclError ret = aclrtMalloc(&inputBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        inputBuffers.push_back(inputBuffer);
        inputSizes.push_back(bufferSize);
    }
    //get model output description and malloc them
    size_t outputSize = m_modelProcess->getNumOutput();;
    std::cout<<"outputSize "<<outputSize<<std::endl;
    for (size_t i = 0; i < outputSize; i++) {
        size_t bufferSize =m_modelProcess->getOutputSizeByIndex(i);
        std::cout<<"i "<<i<<" bufferSize  "<<bufferSize<<std::endl;;
        void *outputBuffer = nullptr;
        //aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
    }
    cout << "finish init AclProcess" << endl;
    return ACL_ERROR_NONE;
}

int AclProcess::Process(Mat& img)
{
    aclError ret = ACL_ERROR_NONE;
    Mat imgResize;
    int batch,channels,height,width;
    aclmdlIODims dims;
    m_modelProcess->GetInputDims(0, dims);
    if(dims.dimCount == 4){
        batch = dims.dims[0];
        height = dims.dims[1];
        width = dims.dims[2];
        channels = dims.dims[3];
    }
    resize(img, imgResize, Size(width, height),INTER_NEAREST);
    aclrtMemcpy(inputBuffers[0], inputSizes[0], imgResize.data,imgResize.cols * imgResize.rows * imgResize.channels(), ACL_MEMCPY_HOST_TO_DEVICE);
    float keypoints[] = {60.,190.,120.,200.,90.,230.,65.,260.,115.,265.,
                        425.,215.,485.,210.,460.,245.,435.,275.,483.,270.,
                        786.,192.,840.,190.,815.,230.,790.,260.,840.,260.,
                        1165.,130.,1225.,130.,1195.,165.,1170.,195.,1215.,195.};

    int32_t face_num = 4;
    std::vector<int64_t> default_keypoint={40,45,72,45,52,65,42,82,72,82};
    std::vector<cv::Point2f> to_point;
    for(int i=0;i<default_keypoint.size()/2;i++){
       to_point.push_back(cv::Point2f(default_keypoint[2*i],default_keypoint[2*i+1]));
    }
    for(int i=0;i<face_num;i++) {
        std::vector<cv::Point2f> from_point;
        for(int j=0;j<to_point.size();j++){
            from_point.push_back(cv::Point2f(keypoints[i*to_point.size()*2+2*j],keypoints[i*to_point.size()*2+2*j+1]));
        }
        cv::Mat trans_M_s = estimateAffine2D(from_point, to_point);
        cv::Mat trans_M=cv::Mat(trans_M_s.size(),CV_32FC1);
        trans_M.at<float>(0,0)=(float)trans_M_s.at<double >(0,0);
        trans_M.at<float>(0,1)=(float)trans_M_s.at<double >(0,1);
        trans_M.at<float>(0,2)=(float)trans_M_s.at<double>(0,2);
        trans_M.at<float>(1,0)=(float)trans_M_s.at<double>(1,0);
        trans_M.at<float>(1,1)=(float)trans_M_s.at<double>(1,1);
        trans_M.at<float>(1,2)=(float)trans_M_s.at<double>(1,2);
        printf(" %f \n",trans_M.at<float>(0,0));
        printf(" %f \n",trans_M.at<float>(0,1));
        printf(" %f \n",trans_M.at<float>(0,2));
        printf(" %f \n",trans_M.at<float>(1,0));
        printf(" %f \n",trans_M.at<float>(1,1));
        printf(" %f \n",trans_M.at<float>(1,2));
        std::cout <<"trans_M.rows "<<trans_M.rows<<" cols "<<trans_M.cols<<" step "<<trans_M.step[0]<<std::endl;
        std::cout <<"inputSizes[1] "<<inputSizes[1]<<" trans_M.step[0]*trans_M.rows "<<trans_M.step[0]*trans_M.rows<<std::endl;
        aclrtMemcpy(inputBuffers[1], inputSizes[1], trans_M.data, trans_M.step[0]*trans_M.rows, ACL_MEMCPY_HOST_TO_DEVICE);
        //forward
        ret = m_modelProcess->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes, stream_);
        if (ret != ACL_ERROR_NONE) {
            cout << "model run faild.ret = " << ret << endl;
            return ret;
        }
        //postprocess
        cout << "begin postprocess" << endl;
        PostProcess(outputBuffers, outputSizes, i, 112, 112);
        cout << "model run success!" << endl;
    }
    return ACL_ERROR_NONE;
}

aclError AclProcess::PostProcess(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes, int face_num, int width, int height)
{
    void* host_data = malloc(outputSizes[0]);
    aclrtMemcpy(host_data, outputSizes[0], outputBuffers[0], outputSizes[0], ACL_MEMCPY_DEVICE_TO_HOST);
    Mat aligned_img = Mat(width, height, CV_8UC3);
    char file_name[20];
    aligned_img.data = ((uchar*)host_data) ;
    sprintf(file_name, "face_aligned_%d.jpg", face_num);
    imwrite(file_name, aligned_img);
    return ACL_ERROR_NONE;
}
