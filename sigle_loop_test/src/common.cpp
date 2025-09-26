#include "common.h"
#include <fstream>
#include <opencv2/core/types.hpp>
#include <sstream>
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
    
    std::cout << "  Processing front image with EigenPlaces..." << std::endl;
    auto front_start = std::chrono::steady_clock::now();
    bool front_success = pModel->ExtractGlobalDescriptor(im, mGlobalDescriptors_front);
    auto front_end = std::chrono::steady_clock::now();
    auto front_time = std::chrono::duration_cast<std::chrono::milliseconds>(front_end - front_start).count();
    
    if (!front_success) {
        std::cerr << "Failed to extract global descriptor from front image" << std::endl;
        mGlobalDescriptors_front = cv::Mat::zeros(1, 2048, CV_32F); // EigenPlaces输出2048维特征
    }
    
    std::cout << "  Processing rear image with EigenPlaces..." << std::endl;
    auto rear_start = std::chrono::steady_clock::now();
    bool rear_success = pModel->ExtractGlobalDescriptor(im_rear, mGlobalDescriptors_rear);
    auto rear_end = std::chrono::steady_clock::now();
    auto rear_time = std::chrono::duration_cast<std::chrono::milliseconds>(rear_end - rear_start).count();
    
    if (!rear_success) {
        std::cerr << "Failed to extract global descriptor from rear image" << std::endl;
        mGlobalDescriptors_rear = cv::Mat::zeros(1, 2048, CV_32F); // EigenPlaces输出2048维特征
    }

    // 将前后帧的全局描述子分别以当前时间戳命名，并保存到
    string front_path = "/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/eigenplaces/front_globaldes/" + to_string(time_stamp) + ".bin";
    string rear_path = "/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/eigenplaces/rear_globaldes/" + to_string(time_stamp) + ".bin";
    cv::FileStorage fs_front(front_path, cv::FileStorage::WRITE);
    fs_front << "global_descriptors" << mGlobalDescriptors_front;
    fs_front.release();
    cv::FileStorage fs_rear(rear_path, cv::FileStorage::WRITE);
    fs_rear << "global_descriptors" << mGlobalDescriptors_rear;
    fs_rear.release();
    string odom_path = "/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/eigenplaces/odom.txt";
    std::ofstream odom_file(odom_path, std::ios::app);
    // 将pose转换为四元数，并保存到odom.txt中,time_stamp保留六位小数
    Eigen::Quaterniond q(pose.block<3,3>(0,0));
    odom_file << to_string(time_stamp) << " " << pose(0,3) << " " << pose(1,3) << " " << pose(2,3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    odom_file.close();
    
    std::cout << "  Front EigenPlaces extraction: " << front_time << "ms, Rear: " << rear_time << "ms" << std::endl;
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
        if (pKF->mPlaceRecognitionScore_front < 0.8){
            candidates_front.push_back(pKF);
        }
        if (pKF->mPlaceRecognitionScore_rear < 0.8) {
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