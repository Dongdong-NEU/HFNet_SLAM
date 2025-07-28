#include "common.h"
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <chrono>
#include <iostream>

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

// KeyFrameNetVlad构造函数实现
KeyFrameNetVlad::KeyFrameNetVlad(int id, const cv::Mat im, BaseModel* pModel, double time_stamp, Eigen::Matrix4d pose) {
    mnFrameId = id;
    timeStamp = time_stamp;
    curPose = pose;
    vector<cv::KeyPoint> vKeyPoints;
    cv::Mat localDescriptors, intermediate;
    pModel->Detect(im, vKeyPoints, localDescriptors, mGlobalDescriptors, 1000, 0.01);
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

// KITTI位姿读取
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

// TUM轨迹读取
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

// TUM轨迹对齐到图像
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

// TUM时间戳获取
vector<double> GetTimesFromFileTUM(const string& filePath, const vector<string>& imageFiles) {
    return AlignTUMTrajectoryToImages(filePath, imageFiles).times;
}

// TUM位姿获取
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFileTUM(const string& filePath, const vector<string>& imageFiles) {
    return AlignTUMTrajectoryToImages(filePath, imageFiles).poses;
}

// OpenCV版本的回环检测
KeyFrameDB GetNCandidateLoopFrameCV(KeyFrameNetVlad* query, const KeyFrameDB &db, int k)
{
    if (db.front()->mnFrameId >= query->mnFrameId - 100) return KeyFrameDB();
    std::vector<KeyFrameNetVlad*> candidates;
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameNetVlad *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 100) break;
        pKF->mPlaceRecognitionScore = cv::norm(query->mGlobalDescriptors - pKF->mGlobalDescriptors, cv::NORM_L2);
        if (pKF->mPlaceRecognitionScore < 0.7)
            candidates.push_back(pKF);
    }
    int num = std::min(k, (int)candidates.size());
    KeyFrameDB res(num);
    std::partial_sort_copy(candidates.begin(), candidates.end(), res.begin(), res.end(),
        [](KeyFrameNetVlad* const f1, KeyFrameNetVlad* const f2) {
            return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
        });
    return res;
}

// Eigen版本的回环检测
KeyFrameDB GetNCandidateLoopFrameEigen(KeyFrameNetVlad* query, const KeyFrameDB &db, int k)
{
    if (db.front()->mnFrameId >= query->mnFrameId - 100) return KeyFrameDB();
    std::vector<KeyFrameNetVlad*> candidates;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
        queryDescriptors(query->mGlobalDescriptors.ptr<float>(), query->mGlobalDescriptors.rows, 
                        query->mGlobalDescriptors.cols);

    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameNetVlad *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 100) break;
        // 位姿平移距离过滤
        Eigen::Vector3d query_t = query->curPose.block<3,1>(0,3);
        Eigen::Vector3d cand_t = pKF->curPose.block<3,1>(0,3);
        double trans_dist = (query_t - cand_t).norm();
        if (trans_dist >= 10.0) continue;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
            pKFDescriptors(
                pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, 
                pKF->mGlobalDescriptors.cols);
        pKF->mPlaceRecognitionScore = (queryDescriptors - pKFDescriptors).norm();
        if (pKF->mPlaceRecognitionScore < 0.8)
            candidates.push_back(pKF);
    }
    int num = std::min(k, (int)candidates.size());
    KeyFrameDB res(num);
    std::partial_sort_copy(candidates.begin(), candidates.end(), res.begin(), res.end(),
        [](KeyFrameNetVlad* const f1, KeyFrameNetVlad* const f2) {
            return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
        });
    if(candidates.size() > 0)
        std::cout << "Candidate size: " << candidates.size() << std::endl;
    return res;
}

// 显示带文本的图像
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
