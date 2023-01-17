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
#include <sys/stat.h>
#include <unistd.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

#include "FCOSDetection.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"
namespace fs = boost::filesystem;
using namespace MxBase;

namespace {
const int RESTENSORF[2] = {100, 5};
const int RESTENSORS[2] = {100, 1};
const int NETINPUTWIDTH = 1333;
const int NETINPUTHEIGHT = 800;
std::string imagePath;
int originImageW;
int originImageH;
float scaleRatio;
int padLeft;
int padRight;
int padTop;
int padBottom;
}  // namespace
/*
这里装载COCO数据集的标签，输入的是标签的路径，最后输出一个标签编号到名称的哈希表。
*/
APP_ERROR FCOSDetection::LoadLabels(const std::string &labelPath,
                                    std::map<int, std::string> &labelMap) {
  std::ifstream infile;
  // open label file
  infile.open(labelPath, std::ios_base::in);
  std::string s;
  // check label file validity
  if (infile.fail()) {
    LogError << "Failed to open label file: " << labelPath << ".";
    return APP_ERR_COMM_OPEN_FAIL;
  }
  labelMap.clear();
  // construct label map
  int count = 0;
  while (std::getline(infile, s)) {
    if (s.find('#') <= 1) {
      continue;
    }
    size_t eraseIndex = s.find_last_not_of("\r\n\t");
    if (eraseIndex != std::string::npos) {
      s.erase(eraseIndex + 1, s.size() - eraseIndex);
    }
    labelMap.insert(std::pair<int, std::string>(count, s));
    count++;
  }
  infile.close();
  return APP_ERR_OK;
}

// 设置模型的初始参数。
void FCOSDetection::SetFCOSPostProcessConfig(
    const InitParam &initParam,
    std::map<std::string, std::shared_ptr<void>> &config) {
  MxBase::ConfigData configData;
  const std::string checkTensor = initParam.checkTensor ? "true" : "false";
  configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
  configData.SetJsonValue("INPUT_TYPE", std::to_string(initParam.inputType));
  configData.SetJsonValue("CHECK_MODEL", checkTensor);
  auto jsonStr = configData.GetCfgJson().serialize();
  config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
  config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
}
// 初始化模型、模型的后处理类以及装载标签。
APP_ERROR FCOSDetection::Init(const InitParam &initParam) {
  deviceId_ = initParam.deviceId;
  APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret != APP_ERR_OK) {
    LogError << "Init devices failed, ret=" << ret << ".";
    return ret;
  }
  ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
  if (ret != APP_ERR_OK) {
    LogError << "Set context failed, ret=" << ret << ".";
    return ret;
  }
  dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
  ret = dvppWrapper_->Init();
  if (ret != APP_ERR_OK) {
    LogError << "DvppWrapper init failed, ret=" << ret << ".";
    return ret;
  }
  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(initParam.modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    return ret;
  }

  std::map<std::string, std::shared_ptr<void>> config;
  SetFCOSPostProcessConfig(initParam, config);
  // init FCOSPostprocess
  post_ = std::make_shared<FCOSPostProcess>();
  ret = post_->Init(config);
  if (ret != APP_ERR_OK) {
    LogError << "FCOSPostprocess init failed, ret=" << ret << ".";
    return ret;
  }
  // load labels from file
  ret = LoadLabels(initParam.labelPath, labelMap_);
  if (ret != APP_ERR_OK) {
    LogError << "Failed to load labels, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}
// 释放占用的内存空间等。
APP_ERROR FCOSDetection::DeInit() {
  dvppWrapper_->DeInit();
  model_->DeInit();
  post_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

/*
这个函数接收图片的路径和一个浮点数组，将读取的图片转化为浮点数组的形式。并且由于整个模型
的输入为CHW，但是opencv读取图片的结果为HWC，所以这里需要进行数据的转换。
*/
APP_ERROR FCOSDetection::ReadImage(const std::string &imgPath,
                                   float *&imageMat) {
  cv::Mat OrigImageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
  if (OrigImageMat.data == NULL) {
    LogInfo << "The image read fail.\n";
    return 0;
  }
  const int IMAGECHANNEL = 3;
  // 对图像进行归一化，均值和方差均为源码提供。
  std::vector<float> std {1.0, 1.0, 1.0};
  std::vector<float> mean {102.9801, 115.9465, 122.7717};
  std::vector<cv::Mat> imageChannels(IMAGECHANNEL);
  cv::split(OrigImageMat, imageChannels);
  for (uint32_t i = 0; i < imageChannels.size(); i++) {
    imageChannels[i].convertTo(imageChannels[i], CV_32FC1, 1.0 / std[i],
                               (0.0 - mean[i]) / std[i]);
  }
  cv::merge(imageChannels, OrigImageMat);
  // 下面这一部分进行数据的前处理，包括resize, 补边。
  originImageW = OrigImageMat.cols;
  originImageH = OrigImageMat.rows;

  scaleRatio = (float)NETINPUTWIDTH * 1.0 / (originImageW * 1.0);
  float cmp = (float)NETINPUTHEIGHT * 1.0 / (originImageH * 1.0);
  if (cmp < scaleRatio) {
    scaleRatio = cmp;
  }
  const int DEVIDE = 2;
  cv::Mat hold;
  int newW = (int)originImageW * scaleRatio;
  int newH = (int)originImageH * scaleRatio;
  cv::resize(OrigImageMat, hold, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
  padLeft = std::max((int)((NETINPUTWIDTH - newW) / DEVIDE), 0);
  padTop = std::max((int)((NETINPUTHEIGHT - newH) / DEVIDE), 0);
  padRight = std::max(NETINPUTWIDTH - newW - padLeft, 0);
  padBottom = std::max(NETINPUTHEIGHT - newH - padTop, 0);
  cv::copyMakeBorder(hold, hold, padTop, padBottom, padLeft, padRight,
                     cv::BORDER_CONSTANT, 0);
  // 这一部分进行HWC到CHW的转换。
  const int CHANNELS = 3;
  std::vector<float> dstData;
  std::vector<cv::Mat> bgrChannels(CHANNELS);
  cv::split(hold, bgrChannels);
  for (uint32_t i = 0; i < bgrChannels.size(); i++) {
    std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
    dstData.insert(dstData.end(), data.begin(), data.end());
  }
  imageMat = (float *)malloc(dstData.size() * sizeof(float));

  for (uint32_t i = 0; i < dstData.size(); i++) {
    imageMat[i] = dstData[i];
  }

  return APP_ERR_OK;
}

/*
这个代码将读取得到的数据转换成为1*3*800*1333的shape的tensor。
*/
APP_ERROR FCOSDetection::CVMatToTensorBase(float *&imageMat,
                                           MxBase::TensorBase &tensorBase) {
  // 整个图片的大小为3*800*1333，又因为数据类型为32位浮点数，所以需要4个字节。故整个所需的存储空间为1*3*800*1333*4。
  const uint32_t dataSize =
      NETINPUTWIDTH * NETINPUTHEIGHT * YUV444_RGB_WIDTH_NU * 4;
  MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
  MemoryData memoryDataSrc(imageMat, dataSize, MemoryData::MEMORY_HOST_MALLOC);
  APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }
  // 构造模型输入的shape=1*3*800*1333，并且最后的数据类型为FLOAT32。
  std::vector<uint32_t> shape = {1, static_cast<uint32_t>(YUV444_RGB_WIDTH_NU),
                                 static_cast<uint32_t>(NETINPUTHEIGHT),
                                 static_cast<uint32_t>(NETINPUTWIDTH)};
  tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);
  return APP_ERR_OK;
}
// 下面这一部分进行模型推理，输入为CVMatToTensorBase函数构造的tensor。
APP_ERROR FCOSDetection::Inference(
    const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs) {
  auto dtypes =
      model_->GetOutputDataType();  // 获取模型的输出张量的数据类型，dtypes[0] =
                                    // 9 - int64; dtypes[1] = 0 - float32。
  auto modelOutputShape =
      model_->GetOutputShape();  // 获取模型的张量输出形状，
                                 // std::vector<std::vector<int64_t>>
  /*
  为整个模型的输出创建张量存储，模型第一维度的张量为1*100的int64类型的张量。
  第二维度为1*100*5的float32类型的张量。
  */
  APP_ERROR ret;
  MxBase::DynamicInfo dynamicInfo = {};
  for (uint32_t i = 0; i < modelOutputShape.size(); i++) {
    std::vector<uint32_t> shapeTensor = {};
    for (uint32_t j = 0; j < modelOutputShape[i].size(); j++) {
      shapeTensor.push_back(modelOutputShape[i][j]);
    }
    MxBase::TensorBase tensor(shapeTensor, dtypes[i],
                              MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                              deviceId_);
    ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
      return ret;
    }
    outputs.push_back(tensor);
  }
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;

  // 模型推理。
  ret = model_->ModelInference(inputs, outputs, dynamicInfo);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

// 对模型推理输出的张量进行后处理。
APP_ERROR FCOSDetection::PostProcess(
    const std::vector<MxBase::TensorBase> &outputs,
    std::vector<std::vector<MxBase::ObjectInfo>> &objInfos) {
  APP_ERROR ret = post_->Process(outputs, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "Process failed, ret = " << ret << ".";
    return ret;
  }

  ret = post_->DeInit();
  if (ret != APP_ERR_OK) {
    LogError << "FCOSPostProcess DeInit failed";
    return ret;
  }
  return APP_ERR_OK;
}
// 将后处理得到的结果写回图片当中。
APP_ERROR FCOSDetection::WriteResult(
    const std::string &imgPath,
    const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos) {
  uint32_t batchSize = objInfos.size();
  std::vector<MxBase::ObjectInfo> resultInfo;
  std::string resPath = imagePath;
  int index = -1;
  const int STRLEN = 4;
  for (uint32_t i = 0; i < resPath.size(); i++) {
    if (resPath.at(i) == '.') {
      resPath.replace(i, STRLEN, ".txt");
      break;
    }
  }
  for (uint32_t i = resPath.size() - 1; i >= 0; i--) {
    if (resPath[i] == '/') {
      index = i;
      break;
    }
  }
  std::string resfile = "./COCORES/";
  resPath = resPath.substr(index + 1, resPath.size() - (index + 1));
  std::ofstream outfile;
  outfile.open(resfile + resPath, std::ios::out);
  for (uint32_t i = 0; i < batchSize; i++) {
    for (uint32_t j = 0; j < objInfos[i].size(); j++) {
      resultInfo.push_back(objInfos[i][j]);
    }
    // 打印置信度最大推理结果
    LogInfo << "result box number is : " << resultInfo.size();
    for (uint32_t j = 0; j < resultInfo.size(); j++) {
      float float_newX0 =
          std::max(((resultInfo[j].x0 - padLeft) / scaleRatio), (float)0);
      float float_newX1 =
          std::max(((resultInfo[j].x1 - padLeft) / scaleRatio), (float)0);
      float float_newY0 =
          std::max(((resultInfo[j].y0 - padTop) / scaleRatio), (float)0);
      float float_newY1 =
          std::max(((resultInfo[j].y1 - padTop) / scaleRatio), (float)0);

      std::string holdStr = std::to_string(resultInfo[j].confidence * 100.0);
      std::string confStr = holdStr.substr(0, holdStr.find(".") + 2 + 1);
      confStr = confStr + "% ";
      // 写入结果
      outfile << float_newX0;
      outfile << " ";
      outfile << float_newY0;
      outfile << " ";
      outfile << float_newX1 - float_newX0;
      outfile << " ";
      outfile << float_newY1 - float_newY0;
      outfile << " ";
      outfile << resultInfo[j].confidence;
      outfile << " ";
      outfile << (int64)resultInfo[j].classId;
      outfile << "\n";
    }
  }

  outfile.close();
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::Process(const std::string &imgPath) {
  imagePath = imgPath;
  float *imageMat;
  // 读取图片，这里输入的参数为图片的路径和存放图片数据的float数组。
  APP_ERROR ret = ReadImage(imagePath, imageMat);
  if (ret != APP_ERR_OK) {
    LogError << "ReadImage failed, ret = " << ret << ".";
    return ret;
  }

  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs = {};
  TensorBase tensorBase;
  // 这里将图片的float数组转换为模型推理输入需要的tensor形式。
  ret = CVMatToTensorBase(imageMat, tensorBase);
  if (ret != APP_ERR_OK) {
    LogError << "CVMatToTensorBase failed, ret = " << ret << ".";
    return ret;
  }
  inputs.push_back(tensorBase);

  // 模型推理，结果存放到outputs中。
  ret = Inference(inputs, outputs);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret = " << ret << ".";
    return ret;
  }
  std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
  // 对模型推理的输出outputs进行后处理，最后输出目标框和置信度。
  ret = PostProcess(outputs, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "PostProcess failed, ret = " << ret << ".";
    return ret;
  }
  // 将得到的目标框和置信度绘制到图片当中。
  ret = WriteResult(imgPath, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "Save result failed, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}