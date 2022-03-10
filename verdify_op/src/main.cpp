/**
* @file Main.cpp
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <iostream>
#include "AclProcess.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    if(argc <= 2){
        cout << "please run: main xxx.om xxx.jpg" <<endl;
        return -1;
    }

    Mat img = imread(argv[2]);
    if(img.empty()){
        cout << "read image faild." << endl;
        return -1;
    }

    AclProcess aclprocess;
    aclError ret = aclprocess.Init(0, argv[1]);
    if(ret != ACL_ERROR_NONE){
        cout << "AclProcess Init faild." << endl;
        return -1;
    }

    aclprocess.Process(img);
    return 0;
}