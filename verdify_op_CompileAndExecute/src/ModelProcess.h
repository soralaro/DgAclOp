/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODELPROCSS_H
#define MODELPROCSS_H

#include <cstdio>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <iostream>
#include <fstream>
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"

using namespace std;

// Class of model inference
class ModelProcess {
public:

    // Construct a new Model Process object for model in the device
    ModelProcess();
    ~ModelProcess();


    int ModelInference(std::vector<void *> &inputBufs, std::vector<size_t> &inputSizes, std::vector<void *> &ouputBufs,
        std::vector<size_t> &outputSizes, aclrtStream stream);
    unsigned int getNumInput(){return 3;}
    unsigned int getInputSizeByIndex(int index){
        if(index==0) {
           return image_shape_[0]*image_shape_[1]*image_shape_[2]*image_shape_[3]*sizeof(unsigned char);
        } else if(index == 1){
           return keypoints_shape_[0]*keypoints_shape_[1]*sizeof(float);
        } else if(index == 2){
           return face_num_shape_[0]* face_num_shape_[1] * sizeof(unsigned int);
        }
        return 0;
    };
    unsigned int getNumOutput(){return 1;}
    unsigned int getOutputSizeByIndex(int index) {
        if(index==0){
           return aligned_image_shape_[0]*aligned_image_shape_[1]*aligned_image_shape_[2]*aligned_image_shape_[3]*sizeof(unsigned char);
        }
        return 0;
    }
    void GetInputDims(int index,aclmdlIODims &dims) {
        if(index==0) {
            dims.dimCount = image_shape_.size();
            dims.dims[0] = image_shape_[0];
            dims.dims[1] = image_shape_[1];
            dims.dims[2] = image_shape_[2];
            dims.dims[3] = image_shape_[3];
        }
        if(index==1) {
            dims.dimCount = keypoints_shape_.size();
            dims.dims[0] = keypoints_shape_[0];
            dims.dims[1] = keypoints_shape_[1];
        }
        if(index==2) {
            dims.dimCount = face_num_shape_.size();
            dims.dims[0] = face_num_shape_[0];
            dims.dims[1] = face_num_shape_[1];
        }
    }
    std::vector<int64_t> image_shape_ = {1, 720, 1280, 3};
    std::vector<int64_t> keypoints_shape_ = {4, 10};
    std::vector<int64_t> face_num_shape_ = {1,1};
    std::vector<int64_t> aligned_image_shape_ = {4, 112, 112, 3};
private:
};

#endif
