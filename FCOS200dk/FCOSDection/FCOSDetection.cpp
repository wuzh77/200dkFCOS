#include "FCOSDetection.h"

#include <sys/stat.h>
#include <unistd.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
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
// load label file.
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

// Set model configuration parameters.
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

APP_ERROR FCOSDetection::DeInit() {
  dvppWrapper_->DeInit();
  model_->DeInit();
  post_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

// read image use opencv and save as Mat.
APP_ERROR FCOSDetection::ReadImage(const std::string &imgPath,
                                   cv::Mat& imageMat) {
  cv::Mat OrigImageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
  if (OrigImageMat.data == NULL) {
    LogInfo << "The image read fail.\n";
    return 0;
  }
  // This part is to preprocess image, resize makeborder.Besides 
  originImageW = OrigImageMat.cols;
  originImageH = OrigImageMat.rows;

  scaleRatio = (float) NETINPUTWIDTH * 1.0 / (originImageW * 1.0);
  float cmp = (float) NETINPUTHEIGHT * 1.0 / (originImageH * 1.0);
  if (cmp < scaleRatio) {
    scaleRatio = cmp;
  }

  int newW = (int) originImageW * scaleRatio;
  int newH = (int) originImageH * scaleRatio;
  cv::resize(OrigImageMat, imageMat, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
  padLeft = std::max((int)((NETINPUTWIDTH - newW) / 2), 0);
  padTop = std::max((int)((NETINPUTHEIGHT - newH) / 2), 0);
  padRight = std::max(NETINPUTWIDTH - newW - padLeft, 0);
  padBottom = std::max(NETINPUTHEIGHT - newH - padTop, 0);
  cv::Mat hold;
  cv::copyMakeBorder(imageMat, hold, padTop, padBottom, padLeft, padRight,
                      cv::BORDER_CONSTANT, 0);
  std::cout << "hold rows = " << hold.rows << " " << "hold cols = " << hold.cols << std::endl;
  int size[] = {3, hold.rows, hold.cols}; // chw - 3 800 1333.
  // transpose image HWC to CHW.
  cv::Mat chw(3, size, CV_32F);
  std::vector<cv::Mat> planes = {
    cv::Mat(hold.rows, hold.cols, CV_32F, hold.ptr(0)),
    cv::Mat(hold.rows, hold.cols, CV_32F, hold.ptr(1)),
    cv::Mat(hold.rows, hold.cols, CV_32F, hold.ptr(2))
  };
  cv::split(hold, planes);
  hold.convertTo(chw, CV_32F);


  imageMat = chw.clone();
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::CVMatToTensorBase(const cv::Mat& imageMat,
                              MxBase::TensorBase& tensorBase)
{
  const uint32_t dataSize = NETINPUTWIDTH * NETINPUTHEIGHT * YUV444_RGB_WIDTH_NU * 4;
  LogInfo << "data Size = " << dataSize;
  MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
  MemoryData memoryDataSrc(imageMat.data, dataSize,
                           MemoryData::MEMORY_HOST_MALLOC);
  APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }
  std::cout << "YUV444_RGB_WIDTH_NU = " << YUV444_RGB_WIDTH_NU << " NETINPUTHEIGHT = " << NETINPUTHEIGHT << " NETINPUTWIDTH = " << NETINPUTWIDTH <<std::endl;
  std::vector<uint32_t> shape = {1, static_cast<uint32_t>(YUV444_RGB_WIDTH_NU), static_cast<uint32_t>(NETINPUTHEIGHT),
                                 static_cast<uint32_t>(NETINPUTWIDTH)};
  tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);
  return APP_ERR_OK;
}
// model reasoning
APP_ERROR FCOSDetection::Inference(
    const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs) {
  auto dtypes = model_->GetOutputDataType();
  LogInfo << "outputDataType0 = " << dtypes[0] << "outputDataType1 = " << dtypes[1];
  // dtypes[0] = 9 - int64; dtypes[1] = 0 - float32
  /* create room for result
   res_tensor[0] is 1*100*1
   res_tensor[1] is 1*100*5 */

  // create for res_tensor[0]
  std::vector<uint32_t> shape1 = {};
  shape1.push_back((uint32_t)RESTENSORS[1]);
  shape1.push_back((uint32_t)RESTENSORS[0]);
  MxBase::TensorBase tensor0(shape1, dtypes[0],
                             MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                             deviceId_);
  APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor0);
  if (ret != APP_ERR_OK) {
    LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
    return ret;
  }
  outputs.push_back(tensor0);

  // create for res_tensor[1]
  std::vector<uint32_t> shape2 = {};
  shape2.push_back(1);
  shape2.push_back((uint32_t)RESTENSORF[0]);
  shape2.push_back((uint32_t)RESTENSORF[1]);
  MxBase::TensorBase tensor1(shape2, dtypes[1],
                             MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                             deviceId_);
  ret = MxBase::TensorBase::TensorBaseMalloc(tensor1);
  if (ret != APP_ERR_OK) {
    LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
    return ret;
  }
  outputs.push_back(tensor1);
  MxBase::DynamicInfo dynamicInfo = {};

  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  LogInfo << "Ready to infer.";
  auto shape = inputs[0].GetShape();

  // print some information of input tensor and model input.
  LogInfo << "input data tensor shape:";
  for (uint32_t i = 0; i<shape.size(); i++) {
    std::cout << " shape" << i << "=" << shape[i];
  }
  std::cout << "\n";
  auto shapeModel = model_->GetInputShape();
  LogInfo << "model real input shape:";
  for (uint32_t i = 0; i<shapeModel.size(); i++) {
    for (uint32_t j = 0; j<shapeModel[i].size(); j++) {
      std::cout << " shape of Model = " << shapeModel[i][j];
    }
  }
  std::cout << std::endl;

  auto dataTypeofModel = model_->GetInputDataType();
  LogInfo << "model real input dataType : ";
  for (uint32_t i = 0; i<dataTypeofModel.size(); i++) {
    std::cout << " input data type = " << dataTypeofModel[i]; 
  }
  std::cout<<std::endl;

  LogInfo << "inputs Type : " << inputs[0].GetDataType();
  
  LogInfo << "model real output dataType : ";
  auto outDataTypeofModel = model_->GetOutputDataType();
  for (uint32_t i = 0; i<outDataTypeofModel.size(); i++) {
    std::cout << "output data type = " << outDataTypeofModel[i];
  }
  std::cout<<std::endl;

  LogInfo << "model output tensor shape : ";
  auto modelOutputShape = model_->GetOutputShape();
  for (uint32_t i = 0; i<modelOutputShape.size(); i++) {
    for (uint32_t j = 0; j<modelOutputShape[i].size(); j++) {
      std::cout << " Output shape of Model = " << modelOutputShape[i][j];
    }
  }
  std::cout << "\n";
  LogInfo << "outputs tensor shape : ";
  for (uint32_t i = 0; i<outputs.size(); i++) {
    auto outputsShape = outputs[i].GetShape();
    for (uint32_t j = 0; j<outputsShape.size(); j++) {
      std::cout << " output shape = " << outputsShape[j];
    }
  }
  std::cout << "\n";

  LogInfo << "inputs Byte size : " << inputs[0].GetByteSize();
  LogInfo << "inputs Size : " << inputs[0].GetSize();
  LogInfo << "inputs GetDataTypeSize : " << inputs[0].GetDataTypeSize();

  auto ModelInputFormat = model_->GetDataFormat();
  LogInfo << "ModelInputFormat = " << ModelInputFormat;
  ret = model_->ModelInference(inputs, outputs, dynamicInfo);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

// postprocess.
APP_ERROR FCOSDetection::PostProcess(
    const std::vector<MxBase::TensorBase> &outputs,
    std::vector<std::vector<MxBase::ObjectInfo>> &objInfos) {
  LogInfo << "start postprocess.\n";
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
  LogInfo << "End to FCOSpostprocess.";
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::WriteResult(
    const std::string& imgPath,
    const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos) {
  cv::Mat imgBgr = cv::imread(imagePath);
  uint32_t batchSize = objInfos.size();
  std::vector<MxBase::ObjectInfo> resultInfo;
  std::string resPath = imagePath;
  int index = -1;
  for (uint32_t i = 0; i<resPath.size(); i++) {
    if (resPath.at(i) == '.') {
      resPath.replace(i, 4, ".txt");
      break;
    }
  }
  for (uint32_t i = resPath.size()-1; i>=0; i--) {
    if (resPath[i] == '/') {
      index = i;
      break;
    }
  }
  std::string resfile = "./COCORES/";
  resPath = resPath.substr(index+1, resPath.size()-(index+1));
  std::ofstream outfile;
  outfile.open(resfile+resPath, std::ios::out);
  for (uint32_t i = 0; i < batchSize; i++) {
    for (uint32_t j = 0; j < objInfos[i].size(); j++) {
      resultInfo.push_back(objInfos[i][j]);
    }
    // 打印置信度最大推理结果
    LogInfo << "result box number is : " << resultInfo.size();
    for (uint32_t j = 0; j < resultInfo.size(); j++) {
      const cv::Scalar green = cv::Scalar(0, 255, 0);
      const cv::Scalar black = cv::Scalar(0, 0, 0);
      const uint32_t thickness = 1;
      const uint32_t lineType = 8;
      const float fontScale = 1.0;

      int newX0 = std::max((int)((resultInfo[j].x0 - padLeft) / scaleRatio), 0);
      int newX1 = std::max((int)((resultInfo[j].x1 - padLeft) / scaleRatio), 0);
      int newY0 = std::max((int)((resultInfo[j].y0 - padTop) / scaleRatio), 0);
      int newY1 = std::max((int)((resultInfo[j].y1 - padTop) / scaleRatio), 0);
      int baseline = 0;
      std::string holdStr = std::to_string(resultInfo[j].confidence * 100.0);
      std::string confStr = holdStr.substr(0, holdStr.find(".") + 2 + 1);
      confStr = confStr + "% ";
      const uint32_t fontFace = cv::FONT_HERSHEY_SCRIPT_COMPLEX;
      cv::Point2i c1(newX0, newY0);
      cv::Point2i c2(newX1, newY1);
      cv::Size sSize = cv::getTextSize(confStr, fontFace, fontScale / 3,
                                       thickness, &baseline);
      cv::Size textSize =
          cv::getTextSize(labelMap_[((int)resultInfo[j].classId)], fontFace,
                          fontScale / 3, thickness, &baseline);
      cv::rectangle(imgBgr, c1,
                    cv::Point(c1.x + textSize.width + 15 + sSize.width,
                              c1.y - textSize.height - 3),
                    green, -1);
      // 在图像上绘制文字
      cv::putText(imgBgr,
                  labelMap_[((int)resultInfo[j].classId)] + ": " + confStr,
                  cv::Point(newX0, newY0 - 2), cv::FONT_HERSHEY_SIMPLEX,
                  fontScale / 3, black, thickness, lineType);
      // 绘制矩形
      cv::rectangle(imgBgr,
                    cv::Rect(newX0, newY0, newX1 - newX0, newY1 - newY0), green,
                    thickness);
      outfile << newX0;
      outfile << " ";
      outfile << newY0;
      outfile << " ";
      outfile << newX1 - newX0;
      outfile << " ";
      outfile << newY1 - newY0;
      outfile << " ";
      outfile << resultInfo[j].confidence;
      outfile << " ";
      outfile << (int)resultInfo[j].classId;
      outfile << "\n";
    }
  }
  outfile.close();
  cv::imwrite("./result.jpg", imgBgr);
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::Process(const std::string &imgPath) {
  imagePath = imgPath;
  cv::Mat imageMat;
  // Read image.
  APP_ERROR ret = ReadImage(imagePath, imageMat);
  if (ret != APP_ERR_OK) {
    LogError << "ReadImage failed, ret = " << ret << ".";
    return ret;
  }

  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs = {};
  TensorBase tensorBase;
  // convert cv::Mat to tensor.
  ret = CVMatToTensorBase(imageMat, tensorBase);
  if (ret != APP_ERR_OK) {
    LogError << "CVMatToTensorBase failed, ret = " << ret << ".";
    return ret;
  }
  inputs.push_back(tensorBase);
  // model infer
  ret = Inference(inputs, outputs);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret = " << ret << ".";
    return ret;
  }
  std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
  // start post process
  ret = PostProcess(outputs, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "PostProcess failed, ret = " << ret << ".";
    return ret;
  }

  ret = WriteResult(imgPath, objInfos);

  if (ret != APP_ERR_OK) {
    LogError << "Save result failed, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}