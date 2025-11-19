#include "common.h"
#include <fstream>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <iomanip>
#include <dirent.h>
#include <iostream>
#include <algorithm>
#include <limits>
#include <string>

// GetPngFiles函数实现
vector<string> GetPngFiles(string strPngDir)
{
    struct dirent **namelist;
    std::vector<std::string> ret;
    int n = scandir(strPngDir.c_str(), &namelist, [](const struct dirent *cur) -> int {
        std::string str(cur->d_name);
        return str.find(".png") != std::string::npos; }, alphasort);

    if (n < 0) {
        return ret;
    }

    for (int i = 0; i < n; i++) {
        std::string filepath(namelist[i]->d_name);
        ret.push_back("/" + filepath);
    }

    free(namelist);
    return ret;
}

cv::Mat CropImage(const cv::Mat& image, int x, int y, int width, int height)
{
    if (image.empty()) {
        std::cerr << "Error: input image is empty!" << std::endl;
        return cv::Mat();
    }

    cv::Rect roi(x, y, width, height);

    roi &= cv::Rect(0, 0, image.cols, image.rows);

    cv::Mat image_cropped = image(roi).clone();
    // cv::resize(image_cropped, image_cropped, cv::Size(600, 450));
      
    return image_cropped;
}
// {
// #include <opencv2/core.hpp>
// #include <iostream>

// cv::Mat LoadGlobalDescriptor(const std::string& path) {
//     cv::FileStorage fs(path, cv::FileStorage::READ);
//     if (!fs.isOpened()) {
//         std::cerr << "Failed to open file: " << path << std::endl;
//         return cv::Mat();
//     }
//     cv::Mat desc;
//     fs["global_descriptors"] >> desc;
//     fs.release();
//     return desc;
// }

// // 用法
// std::string front_path = "..."; // 你的bin文件路径
// cv::Mat global_desc = LoadGlobalDescriptor(front_path);
// if (!global_desc.empty()) {
//     std::cout << "desc shape: " << global_desc.rows << " x " << global_desc.cols << std::endl;
// }
// }

KeyFrameNetVlad::KeyFrameNetVlad(int id, const cv::Mat im, const cv::Mat im_rear, EigenPlacesExtractor* pModel, double time_stamp, Eigen::Matrix4d pose) {
    mnFrameId = id;
    timeStamp = time_stamp;
    curPose = pose;
    
    // std::cout << "  Processing front image with EigenPlaces..." << std::endl;
    auto front_start = std::chrono::steady_clock::now();
    bool front_success = pModel->ExtractGlobalDescriptor(im, mGlobalDescriptors_front);
    auto front_end = std::chrono::steady_clock::now();
    auto front_time = std::chrono::duration_cast<std::chrono::milliseconds>(front_end - front_start).count();
    
    if (!front_success) {
        std::cerr << "Failed to extract global descriptor from front image" << std::endl;
        mGlobalDescriptors_front = cv::Mat::zeros(1, 2048, CV_32F); // EigenPlaces输出2048维特征
    }
    
    // std::cout << "  Processing rear image with EigenPlaces..." << std::endl;
    auto rear_start = std::chrono::steady_clock::now();
    bool rear_success = pModel->ExtractGlobalDescriptor(im_rear, mGlobalDescriptors_rear);
    auto rear_end = std::chrono::steady_clock::now();
    auto rear_time = std::chrono::duration_cast<std::chrono::milliseconds>(rear_end - rear_start).count();
    
    if (!rear_success) {
        std::cerr << "Failed to extract global descriptor from rear image" << std::endl;
        mGlobalDescriptors_rear = cv::Mat::zeros(1, 2048, CV_32F); // EigenPlaces输出2048维特征
    }

    // // 将前后帧的全局描述子分别以当前时间戳命名，并保存到
    // string front_path = "/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/eigenplaces/front_globaldes/" + to_string(time_stamp) + ".bin";
    // string rear_path = "/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/eigenplaces/rear_globaldes/" + to_string(time_stamp) + ".bin";
    // cv::FileStorage fs_front(front_path, cv::FileStorage::WRITE);
    // fs_front << "global_descriptors" << mGlobalDescriptors_front;
    // fs_front.release();
    // cv::FileStorage fs_rear(rear_path, cv::FileStorage::WRITE);
    // fs_rear << "global_descriptors" << mGlobalDescriptors_rear;
    // fs_rear.release();
    // string odom_path = "/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/eigenplaces/odom.txt";
    // std::ofstream odom_file(odom_path, std::ios::app);
    // // 将pose转换为四元数，并保存到odom.txt中,time_stamp保留六位小数
    // Eigen::Quaterniond q(pose.block<3,3>(0,0));
    // odom_file << to_string(time_stamp) << " " << pose(0,3) << " " << pose(1,3) << " " << pose(2,3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    // odom_file.close();
    
    // 控制打印频率：每处理 50 帧打印一次
    static int frame_count = 0;
    frame_count++;
    if (frame_count % 100 == 0) {
        std::cout << "  [Frame " << frame_count << "] Front: " << front_time << "ms, Rear: " << rear_time << "ms" << std::endl;
    }
}
// 定位用
KeyFrameNetVlad::KeyFrameNetVlad(int id ,const cv::Mat im, EigenPlacesExtractor* pModel, double time_stamp, Eigen::Matrix4d pose) {
    mnFrameId = id;
    timeStamp = time_stamp;
    curPose = pose;
    
    bool success = pModel->ExtractGlobalDescriptor(im, mGlobalDescriptors_front);
    if (!success) {
        std::cerr << "Failed to extract global descriptor from image" << std::endl;
        mGlobalDescriptors_front = cv::Mat::zeros(1, 2048, CV_32F); // EigenPlaces输出2048维特征
    }
}

// KITTI时间戳读取
vector<double> GetTimesFromFileKitti(const string& filePath) {
    vector<double> times;
    std::ifstream fin(filePath);
    if (!fin.is_open()) return times;
    string line;
    while (std::getline(fin, line)) {
        times.push_back(std::stod(line));
    }
    return times;
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFileKitti(const string& filePath) {
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    std::ifstream fin(filePath);
    if (!fin.is_open()) return poses;
    string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (vals.size() != 12) continue;
        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
        mat(0,0) = vals[0]; mat(0,1) = vals[1]; mat(0,2) = vals[2];  mat(0,3) = vals[3];
        mat(1,0) = vals[4]; mat(1,1) = vals[5]; mat(1,2) = vals[6];  mat(1,3) = vals[7];
        mat(2,0) = vals[8]; mat(2,1) = vals[9]; mat(2,2) = vals[10]; mat(2,3) = vals[11];
        poses.push_back(mat);
    }
    return poses;
}

std::vector<TUMTrajectoryData, Eigen::aligned_allocator<TUMTrajectoryData>> ReadTUMTrajectory(const string& filePath) {
    std::vector<TUMTrajectoryData, Eigen::aligned_allocator<TUMTrajectoryData>> trajectory;
    std::ifstream fin(filePath);
    if (!fin.is_open()) {
        std::cerr << "Cannot open file: " << filePath << std::endl;
        return trajectory;
    }
    
    string line;
    while (std::getline(fin, line)) {
        // 跳过注释行和空行
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::vector<double> vals;
        double val;
        while (iss >> val) {
            vals.push_back(val);
        }
        
        // TUM格式: timestamp tx ty tz qx qy qz qw (8个值)
        if (vals.size() < 8) continue;
        
        TUMTrajectoryData data;
        data.timestamp = vals[0];
        
        // 提取平移和四元数
        double tx = vals[1], ty = vals[2], tz = vals[3];
        double qx = vals[4], qy = vals[5], qz = vals[6], qw = vals[7];
        
        // 将四元数转换为旋转矩阵
        Eigen::Quaterniond q(qw, qx, qy, qz);
        q.normalize(); // 确保四元数归一化
        Eigen::Matrix3d R = q.toRotationMatrix();
        
        // 保存原始数据
        data.translation = Eigen::Vector3d(tx, ty, tz);
        data.quaternion = q;
        
        // 构建4x4变换矩阵
        data.pose = Eigen::Matrix4d::Identity();
        data.pose.block<3,3>(0,0) = R;
        data.pose(0,3) = tx;
        data.pose(1,3) = ty;
        data.pose(2,3) = tz;
        
        trajectory.push_back(data);
    }
    return trajectory;
}

AlignedTUMData AlignTUMTrajectoryToImages(const string& filePath, const vector<string>& imageFiles) {
    AlignedTUMData result;
    
    // 读取完整轨迹数据
    auto trajectory = ReadTUMTrajectory(filePath);
    if (trajectory.empty()) {
        std::cerr << "Failed to read trajectory from: " << filePath << std::endl;
        return result;
    }
    
    // 为每个图像文件找到最接近的轨迹数据
    for (const string& imageFile : imageFiles) {
        // 从文件名提取时间戳（去掉路径前缀和.png扩展名）
        string filename = imageFile;
        if (filename[0] == '/') {
            filename = filename.substr(1);
        }
        string timestampStr = filename.substr(0, filename.find_last_of('.'));
        double imageTimestamp;
        try {
            imageTimestamp = std::stod(timestampStr);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Cannot parse timestamp from filename: " << imageFile 
                      << " (extracted: '" << timestampStr << "')" << std::endl;
            // 添加默认值
            result.times.push_back(0.0);
            result.poses.push_back(Eigen::Matrix4d::Identity());
            continue;
        }
        
        // 找到最接近的轨迹数据
        double minTimeDiff = std::numeric_limits<double>::max();
        int bestIdx = -1;
        
        for (size_t i = 0; i < trajectory.size(); ++i) {
            double timeDiff = std::abs(trajectory[i].timestamp - imageTimestamp);
            if (timeDiff < minTimeDiff) {
                minTimeDiff = timeDiff;
                bestIdx = i;
            }
        }
        
        if (bestIdx >= 0 && minTimeDiff < 0.1) { // 时间差小于0.1秒才认为匹配
            result.times.push_back(trajectory[bestIdx].timestamp);
            result.poses.push_back(trajectory[bestIdx].pose);
        } else {
            std::cerr << "Warning: No matching data found for image " << imageFile 
                      << " (timestamp: " << imageTimestamp << ")" << std::endl;
            // 使用图像文件名的时间戳和单位矩阵
            result.times.push_back(imageTimestamp);
            result.poses.push_back(Eigen::Matrix4d::Identity());
        }
    }
    
    return result;
}

vector<double> GetTimesFromFileTUM(const string& filePath, const vector<string>& imageFiles) {
    return AlignTUMTrajectoryToImages(filePath, imageFiles).times;
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFileTUM(const string& filePath, const vector<string>& imageFiles) {
    return AlignTUMTrajectoryToImages(filePath, imageFiles).poses;
}

AlignedDualCameraData AlignDualCameraDataToTrajectory(const string& strDatasetPath_front, 
                                                      const string& strDatasetPath_rear, 
                                                      const string& strGTPosesPath) {
    AlignedDualCameraData result;
    
    const vector<string> files_front_raw = GetPngFiles(strDatasetPath_front);
    const vector<string> files_rear_raw = GetPngFiles(strDatasetPath_rear);
    
    if (files_front_raw.empty() || files_rear_raw.empty()) {
        std::cerr << "Error: No images found in front or rear camera directory!" << std::endl;
        return result;
    }
    
    auto trajectory = ReadTUMTrajectory(strGTPosesPath);
    if (trajectory.empty()) {
        std::cerr << "Failed to read trajectory from: " << strGTPosesPath << std::endl;
        return result;
    }
    
    std::vector<std::pair<double, string>> front_timestamps;
    for (const string& imageFile : files_front_raw) {
        string filename = imageFile;
        if (filename[0] == '/') {
            filename = filename.substr(1);
        }
        string timestampStr = filename.substr(0, filename.find_last_of('.'));
        try {
            double timestamp = std::stod(timestampStr);
            front_timestamps.push_back({timestamp, imageFile});
        } catch (const std::invalid_argument& e) {
            continue;
        }
    }
    
    std::vector<std::pair<double, string>> rear_timestamps;
    for (const string& imageFile : files_rear_raw) {
        string filename = imageFile;
        if (filename[0] == '/') {
            filename = filename.substr(1);
        }
        string timestampStr = filename.substr(0, filename.find_last_of('.'));
        try {
            double timestamp = std::stod(timestampStr);
            rear_timestamps.push_back({timestamp, imageFile});
        } catch (const std::invalid_argument& e) {
            continue;
        }
    }
    
    std::sort(front_timestamps.begin(), front_timestamps.end());
    std::sort(rear_timestamps.begin(), rear_timestamps.end());
    
    std::cout << "Processing " << front_timestamps.size() << " front images and " 
              << rear_timestamps.size() << " rear images" << std::endl;

    const double TIME_THRESHOLD = 0.001;
    
    for (const auto& front_pair : front_timestamps) {
        double front_timestamp = front_pair.first;
        string front_file = front_pair.second;
        
        double min_time_diff = std::numeric_limits<double>::max();
        string best_rear_file = "";
        
        for (const auto& rear_pair : rear_timestamps) {
            double rear_timestamp = rear_pair.first;
            double time_diff = std::abs(front_timestamp - rear_timestamp);
            
            if (time_diff < min_time_diff) {
                min_time_diff = time_diff;
                best_rear_file = rear_pair.second;
            }
        }
        
        // 如果找到匹配的后视图像（时间差在阈值内）
        if (min_time_diff < TIME_THRESHOLD && !best_rear_file.empty()) {
            // 寻找最接近的轨迹数据
            double min_traj_diff = std::numeric_limits<double>::max();
            int best_traj_idx = -1;
            
            for (size_t i = 0; i < trajectory.size(); ++i) {
                double traj_diff = std::abs(trajectory[i].timestamp - front_timestamp);
                if (traj_diff < min_traj_diff) {
                    min_traj_diff = traj_diff;
                    best_traj_idx = i;
                }
            }
            
            // 如果轨迹匹配良好（0.1秒内）
            if (best_traj_idx >= 0 && min_traj_diff < 0.1) {
                result.files_front.push_back(front_file);
                result.files_rear.push_back(best_rear_file);
                result.times.push_back(trajectory[best_traj_idx].timestamp);
                result.poses.push_back(trajectory[best_traj_idx].pose);
            }
        }
    }
    
    std::cout << "Successfully aligned " << result.files_front.size() << " image pairs with trajectory data" << std::endl;
    std::cout << "Original front images: " << files_front_raw.size() 
              << ", rear images: " << files_rear_raw.size() << std::endl;
    
    return result;
}


KeyFrameDB GetNCandidateLoopFrameEigen(KeyFrameNetVlad* query, const KeyFrameDB &db, int k, bool &use_rear)
{
    // 当前数据库中的全局描述子最少需要100帧
    if (db.front()->mnFrameId >= query->mnFrameId - 100) return KeyFrameDB();

    std::vector<KeyFrameNetVlad*> candidates;
    std::vector<KeyFrameNetVlad*> candidates_front;
    std::vector<KeyFrameNetVlad*> candidates_rear;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
        queryDescriptors(query->mGlobalDescriptors_front.ptr<float>(), query->mGlobalDescriptors_front.rows, 
                        query->mGlobalDescriptors_front.cols);

    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameNetVlad *pKF = *it;
        // 候选帧的id需要至少与查询帧的id靠后300帧
        if (pKF->mnFrameId > query->mnFrameId - 300) break;
        // 位姿平移距离过滤
        Eigen::Vector3d query_t = query->curPose.block<3,1>(0,3);
        Eigen::Vector3d cand_t = pKF->curPose.block<3,1>(0,3);
        double trans_dist = (query_t - cand_t).norm();
        // 当前帧与候选帧的位姿平移距离需要小于20米
        if (trans_dist >= 20.0) continue;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
            pKFDescriptors_front(
                pKF->mGlobalDescriptors_front.ptr<float>(), pKF->mGlobalDescriptors_front.rows, 
                pKF->mGlobalDescriptors_front.cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
            pKFDescriptors_rear(
                pKF->mGlobalDescriptors_rear.ptr<float>(), pKF->mGlobalDescriptors_rear.rows, 
                pKF->mGlobalDescriptors_rear.cols);
        pKF->mPlaceRecognitionScore_front = (queryDescriptors - pKFDescriptors_front).norm();
        pKF->mPlaceRecognitionScore_rear = (queryDescriptors - pKFDescriptors_rear).norm();
        std::cout << "pKF->mPlaceRecognitionScore_front: " << pKF->mPlaceRecognitionScore_front << std::endl;
        std::cout << "pKF->mPlaceRecognitionScore_rear: " << pKF->mPlaceRecognitionScore_rear << std::endl;
        if (pKF->mPlaceRecognitionScore_front < 0.7){
            candidates_front.push_back(pKF);
        }
        if (pKF->mPlaceRecognitionScore_rear < 0.9) {
            candidates_rear.push_back(pKF);
        }
    }

    if(candidates_front.size() > 0 && candidates_rear.size() == 0) {
        std::cout << "Find front candidates found for query frame: " << query->mnFrameId << " candidates size: " << candidates_front.size() << std::endl;
        int num = std::min(k, (int)candidates_front.size());
        KeyFrameDB res(num);
        std::partial_sort_copy(candidates_front.begin(), candidates_front.end(), res.begin(), res.end(),
            [](KeyFrameNetVlad* const f1, KeyFrameNetVlad* const f2) {
                return f1->mPlaceRecognitionScore_front < f2->mPlaceRecognitionScore_front;
            });
        return res;
    } else if(candidates_front.size() == 0 && candidates_rear.size() > 0) {
        std::cout << "Find rear candidates found for query frame: " << query->mnFrameId << " candidates size: " << candidates_rear.size() << std::endl;
        int num = std::min(k, (int)candidates_rear.size());
        KeyFrameDB res(num);
        std::partial_sort_copy(candidates_rear.begin(), candidates_rear.end(), res.begin(), res.end(),
            [](KeyFrameNetVlad* const f1, KeyFrameNetVlad* const f2) {
                return f1->mPlaceRecognitionScore_rear < f2->mPlaceRecognitionScore_rear;
            });
        use_rear = true; 
        return res;
    } else if(candidates_rear.size() == 0 && candidates_front.size() == 0) {
        std::cout << "No candidates found for query frame: " << query->mnFrameId << std::endl;
        return KeyFrameDB();
    }else if(candidates_rear.size() > 0 && candidates_front.size() > 0) {
        std::cout << "FATAL ERROR: Find front and rear candidates found for query frame: " << query->mnFrameId 
                  << " front size: " << candidates_front.size() 
                  << ", rear size: " << candidates_rear.size() << std::endl;
    }

    return KeyFrameDB();
}

void ShowImageWithText(const string &title, const cv::Mat &image, const string &str)
{
    cv::Mat plot;
    if (image.channels() == 1)
        cv::cvtColor(image, plot, cv::COLOR_GRAY2RGB);
    else
        plot = image.clone();
    cv::putText(plot, str, cv::Point2d(0, 30),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::imshow(title, plot);
}

void LoadConfigYaml(const string &configPath, cv::Size &ImSize) {
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << configPath << std::endl;
        return;
    }
    
    int height = (int)fs["height"];
    int width = (int)fs["width"];
    
    if (height <= 0 || width <= 0) {
        std::cerr << "Invalid height or width in config file." << std::endl;
        return;
    }
    
    ImSize = cv::Size(width, height);
    fs.release();
}

// EigenPlaces模型初始化函数实现
EigenPlacesExtractor* InitEigenPlacesModel(const std::string& onnx_model_path, const std::string& engine_cache_path, cv::Size ImSize) {
    std::cout << "Initializing EigenPlaces model..." << std::endl;
    std::cout << "ONNX model path: " << onnx_model_path << std::endl;
    if (!engine_cache_path.empty()) {
        std::cout << "Engine cache path: " << engine_cache_path << std::endl;
    }

    std::cout << "Input image size: " << ImSize.width << "x" << ImSize.height << std::endl;
    EigenPlacesExtractor* pModel = new EigenPlacesExtractor(onnx_model_path, ImSize, engine_cache_path);
    
    if (!pModel->Initialize()) {
        std::cerr << "Failed to initialize EigenPlaces model" << std::endl;
        delete pModel;
        exit(-1);
    }
    
    if (pModel->IsValid()) {
        std::cout << "Successfully loaded EigenPlaces TensorRT model." << std::endl;
    } else {
        std::cerr << "EigenPlaces model is not valid" << std::endl;
        delete pModel;
        exit(-1);
    }
    
    return pModel;
}

// 从离线描述子文件加载的构造函数实现（双相机）
KeyFrameNetVlad::KeyFrameNetVlad(int id, double time_stamp, Eigen::Matrix4d pose, const string& descriptor_path) {
    mnFrameId = id;
    timeStamp = time_stamp;
    curPose = pose;
    
    // 根据时间戳构建.bin文件路径
    // 格式: save_nn_internal_output_perception_subgraph__vpr_head_nn_output_0_2x2048_f32_<timestamp>.bin
    // 时间戳格式：原始时间戳 * 10，保留1位小数精度 (例如: 1747822563.93 -> 17478225639)
    std::ostringstream oss;
    long long timestamp_int = static_cast<long long>(time_stamp * 10.0);
    oss << timestamp_int;
    string timestamp_str = oss.str();
    string bin_file = descriptor_path + "/save_nn_internal_output_perception_subgraph__vpr_head_nn_output_0_2x2048_f32_" + timestamp_str + ".bin";
    
    // 加载离线描述子
    if (!LoadOfflineDescriptor(bin_file, mGlobalDescriptors_front, mGlobalDescriptors_rear)) {
        std::cerr << "Failed to load offline descriptor from: " << bin_file << std::endl;
        // 使用零矩阵作为默认值
        mGlobalDescriptors_front = cv::Mat::zeros(1, 2048, CV_32F);
        mGlobalDescriptors_rear = cv::Mat::zeros(1, 2048, CV_32F);
    } else {
        std::cout << "Successfully loaded offline descriptor for timestamp: " << time_stamp << std::endl;
    }
}

// 从离线描述子文件加载的构造函数实现（单相机，仅前目）
KeyFrameNetVlad::KeyFrameNetVlad(int id, double time_stamp, Eigen::Matrix4d pose, const string& descriptor_path, bool front_only) {
    mnFrameId = id;
    timeStamp = time_stamp;
    curPose = pose;
    
    // 根据时间戳构建.bin文件路径
    // 格式: save_nn_internal_output_perception_subgraph__vpr_head_nn_output_0_2x2048_f32_<timestamp>.bin
    // 时间戳格式：原始时间戳 * 10，保留1位小数精度 (例如: 1747822563.93 -> 17478225639)
    std::ostringstream oss;
    long long timestamp_int = static_cast<long long>(time_stamp * 10.0);
    oss << timestamp_int;
    string timestamp_str = oss.str();
    string bin_file = descriptor_path + "/save_nn_internal_output_perception_subgraph__vpr_head_nn_output_0_2x2048_f32_" + timestamp_str + ".bin";
    
    // 加载离线描述子
    cv::Mat temp_rear;
    if (!LoadOfflineDescriptor(bin_file, mGlobalDescriptors_front, temp_rear)) {
        std::cerr << "Failed to load offline descriptor from: " << bin_file << std::endl;
        // 使用零矩阵作为默认值
        mGlobalDescriptors_front = cv::Mat::zeros(1, 2048, CV_32F);
    } else {
        std::cout << "Successfully loaded offline descriptor (front only) for timestamp: " << time_stamp << std::endl;
    }
    // 单相机模式不使用后目描述子
}

// 从地图文件加载的构造函数实现（直接设置所有成员变量，不进行特征提取）
KeyFrameNetVlad::KeyFrameNetVlad(int id, double time_stamp, Eigen::Matrix4d pose, 
                                const cv::Mat& desc_front, const cv::Mat& desc_rear) {
    mnFrameId = id;
    timeStamp = time_stamp;
    curPose = pose;
    mGlobalDescriptors_front = desc_front.clone();
    mGlobalDescriptors_rear = desc_rear.clone();
    mPlaceRecognitionScore_front = 1.0;
    mPlaceRecognitionScore_rear = 1.0;
}

// 离线描述子加载函数实现
bool LoadOfflineDescriptor(const string& bin_file_path, cv::Mat& descriptor_front, cv::Mat& descriptor_rear) {
    std::ifstream file(bin_file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open bin file: " << bin_file_path << std::endl;
        return false;
    }
    
    // 读取2x2048的float32数据
    const int batch_size = 2;
    const int feature_dim = 2048;
    const int total_floats = batch_size * feature_dim;
    
    std::vector<float> data(total_floats);
    file.read(reinterpret_cast<char*>(data.data()), total_floats * sizeof(float));
    
    if (!file) {
        std::cerr << "Failed to read data from bin file: " << bin_file_path << std::endl;
        file.close();
        return false;
    }
    file.close();
    
    // batch0是前目图像的全局描述子
    descriptor_front = cv::Mat(1, feature_dim, CV_32F);
    memcpy(descriptor_front.ptr<float>(), data.data(), feature_dim * sizeof(float));
    
    // batch1是后目的全局描述子
    descriptor_rear = cv::Mat(1, feature_dim, CV_32F);
    memcpy(descriptor_rear.ptr<float>(), data.data() + feature_dim, feature_dim * sizeof(float));
    
    return true;
}

// ============================================================================
// 地图保存和加载功能实现
// ============================================================================

/**
 * 保存关键帧数据库到文件
 * 文件格式（二进制）：
 * - 魔数（4字节）："KFDB"
 * - 版本号（4字节）：当前为1
 * - 关键帧数量（4字节）
 * - 对于每个关键帧：
 *   - mnFrameId (int)
 *   - timeStamp (double)
 *   - curPose (16个double, 4x4矩阵)
 *   - front descriptor shape: rows, cols (2个int)
 *   - front descriptor data (rows*cols个float)
 *   - rear descriptor shape: rows, cols (2个int)
 *   - rear descriptor data (rows*cols个float)
 */
bool SaveKeyFrameDatabase(const string& map_file_path, const KeyFrameDB& keyframe_db) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "保存关键帧数据库到: " << map_file_path << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::ofstream file(map_file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法创建地图文件: " << map_file_path << std::endl;
        return false;
    }
    
    // 写入魔数
    const char magic[4] = {'K', 'F', 'D', 'B'};
    file.write(magic, 4);
    
    // 写入版本号
    int version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    
    // 写入关键帧数量
    int num_keyframes = static_cast<int>(keyframe_db.size());
    file.write(reinterpret_cast<const char*>(&num_keyframes), sizeof(int));
    
    std::cout << "正在保存 " << num_keyframes << " 个关键帧..." << std::endl;
    
    // 写入每个关键帧
    int progress = 0;
    for (const auto* kf : keyframe_db) {
        // 帧ID
        file.write(reinterpret_cast<const char*>(&kf->mnFrameId), sizeof(int));
        
        // 时间戳
        file.write(reinterpret_cast<const char*>(&kf->timeStamp), sizeof(double));
        
        // 位姿（4x4矩阵，按行优先存储）
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double val = kf->curPose(i, j);
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
        
        // 前置相机描述子
        int front_rows = kf->mGlobalDescriptors_front.rows;
        int front_cols = kf->mGlobalDescriptors_front.cols;
        file.write(reinterpret_cast<const char*>(&front_rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&front_cols), sizeof(int));
        
        if (kf->mGlobalDescriptors_front.isContinuous()) {
            file.write(reinterpret_cast<const char*>(kf->mGlobalDescriptors_front.data), 
                      front_rows * front_cols * sizeof(float));
        } else {
            for (int i = 0; i < front_rows; ++i) {
                file.write(reinterpret_cast<const char*>(kf->mGlobalDescriptors_front.ptr<float>(i)), 
                          front_cols * sizeof(float));
            }
        }
        
        // 后置相机描述子
        int rear_rows = kf->mGlobalDescriptors_rear.rows;
        int rear_cols = kf->mGlobalDescriptors_rear.cols;
        file.write(reinterpret_cast<const char*>(&rear_rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&rear_cols), sizeof(int));
        
        if (kf->mGlobalDescriptors_rear.isContinuous()) {
            file.write(reinterpret_cast<const char*>(kf->mGlobalDescriptors_rear.data), 
                      rear_rows * rear_cols * sizeof(float));
        } else {
            for (int i = 0; i < rear_rows; ++i) {
                file.write(reinterpret_cast<const char*>(kf->mGlobalDescriptors_rear.ptr<float>(i)), 
                          rear_cols * sizeof(float));
            }
        }
        
        // 进度显示
        progress++;
        if (progress % 100 == 0 || progress == num_keyframes) {
            std::cout << "  进度: " << progress << "/" << num_keyframes 
                     << " (" << (progress * 100 / num_keyframes) << "%)" << std::endl;
        }
    }
    
    file.close();
    
    // 获取文件大小
    std::ifstream file_check(map_file_path, std::ios::binary | std::ios::ate);
    std::streamsize file_size = file_check.tellg();
    file_check.close();
    
    std::cout << "\n地图保存成功！" << std::endl;
    std::cout << "  文件路径: " << map_file_path << std::endl;
    std::cout << "  文件大小: " << (file_size / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  关键帧数: " << num_keyframes << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return true;
}

/**
 * 从文件加载关键帧数据库
 */
bool LoadKeyFrameDatabase(const string& map_file_path, KeyFrameDB& keyframe_db, 
                         std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& keyframe_positions) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "从文件加载关键帧数据库: " << map_file_path << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::ifstream file(map_file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开地图文件: " << map_file_path << std::endl;
        return false;
    }
    
    // 检查魔数
    char magic[4];
    file.read(magic, 4);
    if (magic[0] != 'K' || magic[1] != 'F' || magic[2] != 'D' || magic[3] != 'B') {
        std::cerr << "错误：不是有效的地图文件（魔数不匹配）" << std::endl;
        file.close();
        return false;
    }
    
    // 读取版本号
    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != 1) {
        std::cerr << "错误：不支持的地图文件版本: " << version << std::endl;
        file.close();
        return false;
    }
    
    // 读取关键帧数量
    int num_keyframes;
    file.read(reinterpret_cast<char*>(&num_keyframes), sizeof(int));
    std::cout << "正在加载 " << num_keyframes << " 个关键帧..." << std::endl;
    
    // 清空现有数据
    keyframe_db.clear();
    keyframe_positions.clear();
    keyframe_db.reserve(num_keyframes);
    keyframe_positions.reserve(num_keyframes);
    
    // 读取每个关键帧
    int progress = 0;
    for (int i = 0; i < num_keyframes; ++i) {
        // 读取帧ID
        int frame_id;
        file.read(reinterpret_cast<char*>(&frame_id), sizeof(int));
        
        // 读取时间戳
        double timestamp;
        file.read(reinterpret_cast<char*>(&timestamp), sizeof(double));
        
        // 读取位姿
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(double));
                pose(row, col) = val;
            }
        }
        
        // 读取前置相机描述子
        int front_rows, front_cols;
        file.read(reinterpret_cast<char*>(&front_rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&front_cols), sizeof(int));
        cv::Mat descriptor_front(front_rows, front_cols, CV_32F);
        file.read(reinterpret_cast<char*>(descriptor_front.data), 
                 front_rows * front_cols * sizeof(float));
        
        // 读取后置相机描述子
        int rear_rows, rear_cols;
        file.read(reinterpret_cast<char*>(&rear_rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&rear_cols), sizeof(int));
        cv::Mat descriptor_rear(rear_rows, rear_cols, CV_32F);
        file.read(reinterpret_cast<char*>(descriptor_rear.data), 
                 rear_rows * rear_cols * sizeof(float));
        
        // 使用专门的构造函数创建关键帧对象（不进行特征提取）
        KeyFrameNetVlad* kf = new KeyFrameNetVlad(frame_id, timestamp, pose, 
                                                  descriptor_front, descriptor_rear);
        
        keyframe_db.push_back(kf);
        keyframe_positions.push_back(pose.block<3,1>(0,3));
        
        // 进度显示
        progress++;
        if (progress % 100 == 0 || progress == num_keyframes) {
            std::cout << "  进度: " << progress << "/" << num_keyframes 
                     << " (" << (progress * 100 / num_keyframes) << "%)" << std::endl;
        }
    }
    
    file.close();
    
    std::cout << "\n地图加载成功！" << std::endl;
    std::cout << "  关键帧数: " << keyframe_db.size() << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return true;
}