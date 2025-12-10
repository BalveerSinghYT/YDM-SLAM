/*
* This file is part of YDM-SLAM
* Author: Balveer Singh
* GitHub: https://github.com/balveersinghyt/YDM-SLAM
*/

#include "yolov8_seg_onnx.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;

/**
 * @brief 读取并初始化ONNX模型
 * @param modelPath 模型文件路径
 * @param isCuda 是否使用CUDA加速
 * @param cudaID CUDA设备ID
 * @param warmUp 是否进行预热运行
 * @return 成功返回true,失败返回false
 */
bool Yolov8SegOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
    // 确保批处理大小至少为1
    if (_batchSize < 1) _batchSize = 1;
    try
    {
        // 检查模型路径是否有效
        if (!CheckModelPath(modelPath))
            return false;
        
        // 获取可用的执行提供程序列表
        std::vector<std::string> available_providers = GetAvailableProviders();
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");

        // 判断是否使用GPU推理
        if (isCuda && (cuda_available == available_providers.end()))
        {
            // CUDA不可用,切换到CPU
            std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        else if (isCuda && (cuda_available != available_providers.end()))
        {
            // 配置CUDA执行提供程序
            std::cout << "************* Infer model on GPU! *************" << std::endl;
            
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = cudaID;
            // 使用启发式搜索代替穷举搜索，避免CUDNN错误
            cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::HEURISTIC;
            cudaOption.arena_extend_strategy = 0;
            cudaOption.do_copy_in_default_stream = 1;
            
            _OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        }
        else
        {
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        
        // 设置图优化级别为扩展优化
        _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // 根据不同操作系统加载模型
#ifdef _WIN32
        std::wstring model_path(modelPath.begin(), modelPath.end());
        _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        
        // 初始化输入节点信息
        _inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
        _inputName = _OrtSession->GetInputName(0, allocator);
        _inputNodeNames.push_back(_inputName);
#else
        _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
        _inputNodeNames.push_back(_inputName.get());
#endif

        // 获取输入张量的类型和形状信息
        Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
        auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
        _inputNodeDataType = input_tensor_info.GetElementType();
        _inputTensorShape = input_tensor_info.GetShape();

        // 处理动态批处理大小(-1表示动态维度)
        if (_inputTensorShape[0] == -1)
        {
            _isDynamicShape = true;
            _inputTensorShape[0] = _batchSize;
        }
        // 处理动态输入尺寸
        if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
            _isDynamicShape = true;
            _inputTensorShape[2] = _netHeight;
            _inputTensorShape[3] = _netWidth;
        }
        
        // 初始化输出节点信息
        _outputNodesNum = _OrtSession->GetOutputCount();
        if (_outputNodesNum != 2) {
            // 分割模型必须有2个输出(检测结果和掩码原型)
            cout << "This model has " << _outputNodesNum << "output, which is not a segmentation model.Please check your model name or path!" << endl;
            return false;
        }
        
        // 获取两个输出节点的名称
#if ORT_API_VERSION < ORT_OLD_VISON
        _output_name0 = _OrtSession->GetOutputName(0, allocator);
        _output_name1 = _OrtSession->GetOutputName(1, allocator);
#else
        _output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
        _output_name1 = std::move(_OrtSession->GetOutputNameAllocated(1, allocator));
#endif
        
        Ort::TypeInfo type_info_output0(nullptr);
        Ort::TypeInfo type_info_output1(nullptr);
        bool flag = false;
        
        // 确保"output0"在"output1"之前(字典序排序)
#if ORT_API_VERSION < ORT_OLD_VISON
        flag = strcmp(_output_name0, _output_name1) < 0;
#else
        flag = strcmp(_output_name0.get(), _output_name1.get()) < 0;
#endif
        if (flag)  
        {
            type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0: 检测结果
            type_info_output1 = _OrtSession->GetOutputTypeInfo(1);  //output1: 掩码原型
#if ORT_API_VERSION < ORT_OLD_VISON
            _outputNodeNames.push_back(_output_name0);
            _outputNodeNames.push_back(_output_name1);
#else
            _outputNodeNames.push_back(_output_name0.get());
            _outputNodeNames.push_back(_output_name1.get());
#endif
        }
        else {
            type_info_output0 = _OrtSession->GetOutputTypeInfo(1);  //output0
            type_info_output1 = _OrtSession->GetOutputTypeInfo(0);  //output1
#if ORT_API_VERSION < ORT_OLD_VISON
            _outputNodeNames.push_back(_output_name1);
            _outputNodeNames.push_back(_output_name0);
#else
            _outputNodeNames.push_back(_output_name1.get());
            _outputNodeNames.push_back(_output_name0.get());
#endif
        }

        // 获取输出张量的类型和形状信息
        auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
        _outputNodeDataType = tensor_info_output0.GetElementType();
        _outputTensorShape = tensor_info_output0.GetShape();
        auto tensor_info_output1 = type_info_output1.GetTensorTypeAndShapeInfo();
        
        // GPU预热:运行几次推理以优化性能
        if (isCuda && warmUp) {
            cout << "Start warming up" << endl;
            size_t input_tensor_length = VectorProduct(_inputTensorShape);
            float* temp = new float[input_tensor_length];
            std::vector<Ort::Value> input_tensors;
            std::vector<Ort::Value> output_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
                _inputTensorShape.size()));
            // 执行3次预热推理
            for (int i = 0; i < 3; ++i) {
                output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                    _inputNodeNames.data(),
                    input_tensors.data(),
                    _inputNodeNames.size(),
                    _outputNodeNames.data(),
                    _outputNodeNames.size());
            }
            delete[]temp;
        }
    }
    catch (const std::exception&) {
        return false;
    }
    return true;
}

/**
 * @brief 图像预处理:调整大小、填充边界
 * @param srcImgs 源图像向量
 * @param outSrcImgs 预处理后的图像向量
 * @param params 预处理参数(缩放比例和偏移)
 * @return 状态码
 */
int Yolov8SegOnnx::PreProcessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
    outSrcImgs.clear();
    Size input_size = Size(_netWidth, _netHeight);
    
    // 对每张图像进行LetterBox处理
    for (int i = 0; i < srcImgs.size(); ++i) {
        Mat temp_img = srcImgs[i];
        Vec4d temp_param = { 1,1,0,0 }; // [ratio_x, ratio_y, offset_x, offset_y]
        if (temp_img.size() != input_size) {
            Mat borderImg;
            // LetterBox: 保持宽高比的缩放和填充
            LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
            outSrcImgs.push_back(borderImg);
            params.push_back(temp_param);
        }
        else {
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }

    // 如果图像数量不是批处理大小的倍数,用零图像填充
    int lack_num = srcImgs.size() % _batchSize;
    if (lack_num != 0) {
        for (int i = 0; i < lack_num; ++i) {
            Mat temp_img = Mat::zeros(input_size, CV_8UC3);
            Vec4d temp_param = { 1,1,0,0 };
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }
    return 0;
}

/**
 * @brief 单张图像检测(封装批处理函数)
 * @param srcImg 输入图像
 * @param output 检测结果输出
 * @return 成功返回true
 */
bool Yolov8SegOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) {
    std::vector<cv::Mat> input_data = { srcImg };
    std::vector<std::vector<OutputParams>> tenp_output;
    if (OnnxBatchDetect(input_data, tenp_output)) {
        output = tenp_output[0];
        return true;
    }
    else return false;
}

/**
 * @brief 批量图像检测和分割
 * @param srcImgs 输入图像向量
 * @param output 每张图像的检测结果向量
 * @return 成功返回true
 */
bool Yolov8SegOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) {
    vector<Vec4d> params;
    vector<Mat> input_images;
    cv::Size input_size(_netWidth, _netHeight);
    
    // 图像预处理
    PreProcessing(srcImgs, input_images, params);
    // 将图像转换为blob格式: [batch, channels, height, width]
    // 归一化到[0,1],BGR转RGB,不进行均值减法
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

    // 创建输入张量
    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

    // 执行推理
    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),
        input_tensors.data(),
        _inputNodeNames.size(),
        _outputNodeNames.data(),
        _outputNodeNames.size()
    );

    // 后处理
    // output0: [batch, 116, 8400] -> 检测框和掩码系数
    // output1: [batch, 32, 160, 160] -> 掩码原型
    float* all_data = output_tensors[0].GetTensorMutableData<float>();
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    
    vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
    int mask_protos_length = VectorProduct(mask_protos_shape);
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
    int net_width = (int)_outputTensorShape[1];
    int socre_array_length = net_width - 4 - _outputMaskTensorShape[1]; // 类别分数数量
    
    // 处理每张图像的检测结果
    for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
        // 转置输出: [116, 8400] -> [8400, 116]
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();
        all_data += one_output_length;
        float* pdata = (float*)output0.data;
        int rows = output0.rows;
        
        // 存储检测结果的容器
        std::vector<int> class_ids;          // 类别ID
        std::vector<float> confidences;       // 置信度
        std::vector<cv::Rect> boxes;          // 边界框
        std::vector<vector<float>> picked_proposals;  // 掩码系数
        
        // 遍历每个检测框
        for (int r = 0; r < rows; ++r) {
            // 提取类别分数: [4+类别数+32] 中的类别部分
            cv::Mat scores(1, socre_array_length, CV_32F, pdata + 4);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            
            // 过滤低置信度检测
            if (max_class_socre >= _classThreshold) {
                // 提取掩码系数(最后32个值)
                vector<float> temp_proto(pdata + 4 + socre_array_length, pdata + net_width);
                picked_proposals.push_back(temp_proto);
                
                // 将边界框坐标从网络空间转换回原始图像空间
                // 格式: [center_x, center_y, width, height]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0];
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1];
                float w = pdata[2] / params[img_index][0];
                float h = pdata[3] / params[img_index][1];
                
                // 转换为左上角坐标格式
                int left = MAX(int(x - 0.5 * w + 0.5), 0);
                int top = MAX(int(y - 0.5 * h + 0.5), 0);
                
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre);
                boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width; // 移动到下一行
        }

        // 非极大值抑制(NMS):去除重叠的检测框
        vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
        
        std::vector<vector<float>> temp_mask_proposals;
        cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
        std::vector<OutputParams> temp_output;
        
        // 收集NMS后保留的检测结果
        for (int i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            OutputParams result;
            result.id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx] & holeImgRect; // 裁剪到图像范围内
            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }

        // 配置掩码生成参数
        MaskParams mask_params;
        mask_params.params = params[img_index];
        mask_params.srcImgShape = srcImgs[img_index].size();
        mask_params.netHeight = _netHeight;
        mask_params.netWidth = _netWidth;
        mask_params.maskThreshold = _maskThreshold;
        
        // 获取当前图像的掩码原型: [1, 32, 160, 160]
        Mat mask_protos = Mat(mask_protos_shape, CV_32F, output_tensors[1].GetTensorMutableData<float>() + img_index * mask_protos_length);
        
        // 为每个检测结果生成分割掩码
        // 通过掩码系数和掩码原型的线性组合生成最终掩码
        for (int i = 0; i < temp_mask_proposals.size(); ++i) {
            GetMask2(Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
        }

        // 旧版本的批量掩码生成方法(已注释)
        // If the GetMask2() still reports errors, it is recommended to use GetMask().
        //Mat mask_proposals;
        //for (int i = 0; i < temp_mask_proposals.size(); ++i) {
        //	mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
        //}
        //GetMask(mask_proposals, mask_protos, temp_output, mask_params);
        
        output.push_back(temp_output);
    }

    if (output.size())
        return true;
    else
        return false;
}