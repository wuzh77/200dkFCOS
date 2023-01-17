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

#include "FCOSDetectionPostProcess.h"

#include <fstream>

#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
namespace {
const uint32_t L = 0;
const uint32_t T = 1;
const uint32_t R = 2;
const uint32_t B = 3;
const int NETINPUTWIDTH = 1333;
const int NETINPUTHEIGHT = 800;
const uint32_t CENTERPOINT = 4;
const float THRESHOLD_ = 0.4;
}  // namespace
using namespace MxBase;

FCOSPostProcess &FCOSPostProcess::operator=(const FCOSPostProcess &other) {
  if (this == &other) {
    return *this;
  }
  ObjectPostProcessBase::operator=(other);
  return *this;
}

APP_ERROR FCOSPostProcess::Init(
    const std::map<std::string, std::shared_ptr<void>> &postConfig) {
  LogInfo << "Start to Init FCOSDetectionPostProcess";
  APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret)
             << "Fail to superInit in FCOSDetectionPostProcess.";
    return ret;
  }
  LogInfo << "End to Init FCOSDetectionPostProcess.";
  return APP_ERR_OK;
}

APP_ERROR FCOSPostProcess::DeInit() { return APP_ERR_OK; }

/*
    input:
        tensors:the output of mxpi_tensorinfer0 , the output of the model.
        objectInfos:save result.
    return:
        return the postprocess result.
*/
APP_ERROR FCOSPostProcess::Process(
    const std::vector<TensorBase> &tensors,
    std::vector<std::vector<ObjectInfo>> &objectInfos,
    const std::vector<ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
  LogInfo << "Start to Process FCOSDetectionPostProcess.";
  APP_ERROR ret = APP_ERR_OK;
  auto inputs = tensors;
  ret = CheckAndMoveTensors(inputs);
  if (ret != APP_ERR_OK) {
    LogError << "CheckAndMoveTensors failed. ret=" << ret;
    return ret;
  }

  LogInfo << "FCOSDetectionPostProcess start to write results.";

  for (auto num : {0, 1}) {
    if (((uint32_t)num >= (uint32_t)tensors.size()) || (num < 0)) {
      LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
               << ") must ben less than tensors'size(" << tensors.size()
               << ") and larger than 0.";
    }
  }
  auto shape = tensors[0].GetShape();
  if (shape.size() == 0) {
    return APP_ERR_OK;
  }
  LogInfo << "start to process.";
  if (tensors[0].GetBuffer() == NULL || tensors[1].GetBuffer() == NULL) {
    LogError << "tensors buffer is NULL.\n";
    return APP_ERR_OK;
  }
  std::vector<ObjectInfo> objectInfo;

  std::vector<uint32_t> shape1;  // shape1 ----- class
  std::vector<uint32_t> shape2;  // shape2 ----- box
  float *res0;
  __int64 *classIdx;  // int64
  const int CHANNELS = 3;
  const int ROWSIZE = 5;
  const int BOXSIZE = 100;
  if (tensors[0].GetShape().size() == CHANNELS) {
    res0 = (float *)tensors[0].GetBuffer();
    classIdx = (__int64 *)tensors[1].GetBuffer();
    shape2 = tensors[0].GetShape();
    shape1 = tensors[1].GetShape();
  } else {
    res0 = (float *)tensors[1].GetBuffer();
    classIdx = (__int64 *)tensors[0].GetBuffer();
    shape2 = tensors[1].GetShape();
    shape1 = tensors[0].GetShape();
  }
  for (uint32_t i = 0; i < shape2[0]; i++) {    // shape2[0] = 1
    for (uint32_t j = 0; j < shape2[1]; j++) {  // shape2[1] = 100
      if (res0[i * 100 + j * ROWSIZE + CENTERPOINT] <= THRESHOLD_) continue;
      ObjectInfo objInfo;
      objInfo.x0 = res0[i * BOXSIZE + j * ROWSIZE + L];
      objInfo.y0 = res0[i * BOXSIZE + j * ROWSIZE + T];
      objInfo.x1 = res0[i * BOXSIZE + j * ROWSIZE + R];
      objInfo.y1 = res0[i * BOXSIZE + j * ROWSIZE + B];
      objInfo.confidence = res0[i * BOXSIZE + j * ROWSIZE + CENTERPOINT];
      objInfo.classId = (float)classIdx[i * BOXSIZE + j];  // 1*100
      objectInfo.push_back(objInfo);
    }
  }
  objectInfos.push_back(objectInfo);
  LogInfo << "FCOSDetectionPostProcess write results successed.";
  LogInfo << "End to Process FCOSDetectionPostProcess.";
  return APP_ERR_OK;
}