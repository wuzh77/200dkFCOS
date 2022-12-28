#ifndef MXBASE_FCOSDETECTION_H
#define MXBASE_FCOSDETECTION_H

#include <FCOSDetectionPostProcess.h>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    bool checkTensor;
    std::string modelPath;
    uint32_t classNum;
    uint32_t inputType;
};

class FCOSDetection {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &outputs,
                          std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR CVMatToTensorBase(const cv::Mat& imageMat,
                              MxBase::TensorBase& tensorBase);
protected:
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat& imageMat);
    APP_ERROR LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap);
    APP_ERROR WriteResult(const std::string& imgPath,
                         const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);
    void SetFCOSPostProcessConfig(const InitParam &initParam, std::map<std::string, std::shared_ptr<void>> &config);
private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_; // 封装DVPP基本编码、解码、扣图功能
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_; // 模型推理功能处理
    std::shared_ptr<FCOSPostProcess> post_;
    MxBase::ModelDesc modelDesc_ = {}; // 模型描述信息
    std::map<int, std::string> labelMap_ = {};
    uint32_t deviceId_ = 0;
};
#endif