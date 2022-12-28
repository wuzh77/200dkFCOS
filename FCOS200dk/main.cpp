/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include <FCOSDetection.h>

#include <iostream>
#include <vector>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "MxBase/Log/Log.h"

namespace {
const uint32_t CLASS_NU = 80;
}
void InitFCOSParam(InitParam& initParam) {
  initParam.deviceId = 0;
  initParam.labelPath = "./models/coco.names";
  initParam.checkTensor = true;
  initParam.modelPath = "/home/dongyu3/FCOS200dk/models/fcos_bs1.om";
  initParam.inputType = 0;
  initParam.classNum = CLASS_NU;
}
void GetFileNames(std::string path, std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (std::strcmp(ptr->d_name, ".") != 0 && std::strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
    }
    }
    closedir(pDir);
}

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    LogWarn << "Please input image path.";
    return APP_ERR_OK;
  }

  InitParam initParam;
  InitFCOSParam(initParam);
  std::printf("Initialize FCOS param successfully.\n");
  auto FCOS = std::make_shared<FCOSDetection>();
  // initialize information of model reasoning.
  APP_ERROR ret = FCOS->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "FCOSDetection initialize failed. ";
    return ret;
  }

  // std::string imgPath = argv[1];
  // start reasoning.
  std::string imageFile = argv[1];
  std::vector<std::string> filename;
  GetFileNames(imageFile, filename);
  for (uint32_t i = 0; i < filename.size(); i++) {
     std::cout << "process image number is " << i << std::endl;
     std::cout << "pic name = " << filename[i] << std::endl;
     ret = FCOS->Process(filename[i]);
  }
  // if (ret != APP_ERR_OK) {
  //   LogError << "FCOSDetection process failed.";
  //   FCOS->DeInit();
  //   return ret;
  // }
  FCOS->DeInit();
  return APP_ERR_OK;
}