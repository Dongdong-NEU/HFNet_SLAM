#ifndef LOOP_VISUAL_H
#define LOOP_VISUAL_H

#include <mutex>
#include <pangolin/pangolin.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>
#include "common.h"

// Pangolin可视化类
class TrajectoryViewer {
private:
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> trajectory_;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> keyframe_positions_;
    Eigen::Vector3d current_position_;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> kdtree_candidates_;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> eigen_candidates_;
    std::vector<float> eigen_scores_;
    int current_frame_id_;
    bool data_updated_;
    bool is_paused_;
    std::mutex data_mutex_;
    
public:
    TrajectoryViewer();
    
    // 获取暂停状态
    bool IsPaused();
    
    // 更新轨迹数据
    void UpdateTrajectory(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses);
    
    // 更新关键帧位置
    void UpdateKeyFrames(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& positions);
    
    // 更新当前查询帧和候选帧
    void UpdateLoopDetection(int frame_id, const Eigen::Vector3d& current_pos,
                           const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& kdtree_cands,
                           const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& eigen_cands,
                           const std::vector<float>& scores);
    
    // 主可视化循环
    void Run();
};

// 全局可视化器实例
extern TrajectoryViewer* g_viewer;

// 函数声明
void StartVisualization();
void UpdateVisualization(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses,
                        int queryFrameId, const KeyFrameDB& loopCandidates,
                        const std::vector<size_t>& kdtreeCandidates, const KeyFrameDB& keyFrameDB);

#endif // LOOP_VISUAL_H
