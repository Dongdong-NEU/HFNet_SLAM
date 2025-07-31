#include "loop_visual.h"

// 全局可视化器实例
TrajectoryViewer* g_viewer = nullptr;

// TrajectoryViewer类实现
TrajectoryViewer::TrajectoryViewer() : current_frame_id_(-1), data_updated_(false), is_paused_(false) {}

// 获取暂停状态
bool TrajectoryViewer::IsPaused() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return is_paused_;
}

// 更新轨迹数据
void TrajectoryViewer::UpdateTrajectory(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // 只有在数据实际变化时才更新
    if (trajectory_.size() != poses.size() || trajectory_ != poses) {
        trajectory_ = poses;
        data_updated_ = true;
    }
}

// 更新关键帧位置
void TrajectoryViewer::UpdateKeyFrames(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& positions) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // 只有在数据实际变化时才更新
    if (keyframe_positions_.size() != positions.size() || keyframe_positions_ != positions) {
        keyframe_positions_ = positions;
        data_updated_ = true;
    }
}

// 更新当前查询帧和候选帧
void TrajectoryViewer::UpdateLoopDetection(int frame_id, const Eigen::Vector3d& current_pos,
                       const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& kdtree_cands,
                       const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& eigen_cands,
                       const std::vector<float>& scores) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_frame_id_ = frame_id;
    current_position_ = current_pos;
    kdtree_candidates_ = kdtree_cands;
    eigen_candidates_ = eigen_cands;
    eigen_scores_ = scores;
    data_updated_ = true;
    
    // 调试输出：确保数据正确更新
    if (!eigen_cands.empty()) {
        // std::cout << "UpdateLoopDetection: Updated with " << eigen_cands.size() 
        //           << " Eigen candidates for frame " << frame_id << std::endl;
    }
}

// 主可视化循环
void TrajectoryViewer::Run() {
    // 创建OpenGL上下文和窗口
    pangolin::CreateWindowAndBind("Detect Loop Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // 设置相机
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    
    // 创建交互式视图
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    
    // 创建控制面板
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    
    // 添加控制变量
    pangolin::Var<bool> menu_show_trajectory("menu.Show Trajectory", true, true);
    pangolin::Var<bool> menu_show_keyframes("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menu_show_current("menu.Show Current", true, true);
    pangolin::Var<bool> menu_show_kdtree("menu.Show KDTree", true, true);
    pangolin::Var<bool> menu_show_eigen("menu.Show Eigen", true, true);
    pangolin::Var<float> menu_point_size("menu.Point Size", 3.0f, 1.0f, 10.0f);
    pangolin::Var<float> menu_line_width("menu.Line Width", 2.0f, 1.0f, 5.0f);
    pangolin::Var<bool> menu_pause("menu.Pause Program", false, true);
    
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        
        // 更新暂停状态
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            is_paused_ = menu_pause;
        }
        
        // 设置背景颜色
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        
        // 获取数据副本（只在数据更新时）
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> traj_copy;
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> kf_copy, kdtree_copy, eigen_copy;
        std::vector<float> scores_copy;
        Eigen::Vector3d current_copy;
        int frame_id_copy;
        bool has_new_data = false;
        
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (data_updated_) {
                traj_copy = trajectory_;
                kf_copy = keyframe_positions_;
                kdtree_copy = kdtree_candidates_;
                eigen_copy = eigen_candidates_;
                scores_copy = eigen_scores_;
                current_copy = current_position_;
                frame_id_copy = current_frame_id_;
                data_updated_ = false;
                has_new_data = true;
            }
        }
        
        // 如果没有新数据，使用上一次的数据
        static std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> last_traj_copy;
        static std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> last_kf_copy, last_kdtree_copy, last_eigen_copy;
        static std::vector<float> last_scores_copy;
        static Eigen::Vector3d last_current_copy;
        static int last_frame_id_copy = -1;
        
        if (has_new_data) {
            last_traj_copy = traj_copy;
            last_kf_copy = kf_copy;
            last_kdtree_copy = kdtree_copy;
            last_eigen_copy = eigen_copy;
            last_scores_copy = scores_copy;
            last_current_copy = current_copy;
            last_frame_id_copy = frame_id_copy;
        } else {
            traj_copy = last_traj_copy;
            kf_copy = last_kf_copy;
            kdtree_copy = last_kdtree_copy;
            eigen_copy = last_eigen_copy;
            scores_copy = last_scores_copy;
            current_copy = last_current_copy;
            frame_id_copy = last_frame_id_copy;
        }
        
        // 绘制轨迹
        if (menu_show_trajectory && !traj_copy.empty()) {
            glColor3f(0.5f, 0.5f, 0.5f);
            glLineWidth(menu_line_width);
            glBegin(GL_LINE_STRIP);
            for (const auto& pose : traj_copy) {
                glVertex3f(pose(0,3), pose(1,3), pose(2,3));
            }
            glEnd();
        }
        
        // 绘制关键帧
        if (menu_show_keyframes && !kf_copy.empty()) {
            glColor3f(0.0f, 1.0f, 0.0f);
            glPointSize(menu_point_size);
            glBegin(GL_POINTS);
            for (const auto& pos : kf_copy) {
                glVertex3f(pos.x(), pos.y(), pos.z());
            }
            glEnd();
        }
        
        // 绘制当前位置
        if (menu_show_current && frame_id_copy >= 0) {
            glColor3f(1.0f, 0.0f, 0.0f);
            glPointSize(menu_point_size * 2);
            glBegin(GL_POINTS);
            glVertex3f(current_copy.x(), current_copy.y(), current_copy.z());
            glEnd();
        }
        
        // 绘制Eigen候选帧（先绘制，避免被KDTree线条覆盖）
        if (menu_show_eigen && !eigen_copy.empty()) {
            // std::cout << "Pangolin: Drawing " << eigen_copy.size() << " Eigen candidates (yellow)" << std::endl;
            // 绘制黄色点
            glColor3f(1.0f, 1.0f, 0.0f);
            glPointSize(menu_point_size * 2.0f);  // 稍大一些以便观察
            glBegin(GL_POINTS);
            for (const auto& pos : eigen_copy) {
                glVertex3f(pos.x(), pos.y(), pos.z());
            }
            glEnd();
            
            // 绘制黄色连线（更粗一些，确保可见）
            if (frame_id_copy >= 0) {
                // std::cout << "Pangolin: Drawing " << eigen_copy.size() << " yellow lines from frame " 
                        //   << frame_id_copy << std::endl;
                glColor3f(1.0f, 1.0f, 0.0f);
                glLineWidth(menu_line_width * 2.0f);  // 更粗的线
                for (const auto& pos : eigen_copy) {
                    glBegin(GL_LINES);
                    glVertex3f(current_copy.x(), current_copy.y(), current_copy.z());
                    glVertex3f(pos.x(), pos.y(), pos.z());
                    glEnd();
                }
            }
        } else if (menu_show_eigen) {
            // std::cout << "Pangolin: No Eigen candidates to draw (empty eigen_copy)" << std::endl;
        }
        
        // 绘制KDTree候选帧（后绘制，使用较细的线）
        if (menu_show_kdtree && !kdtree_copy.empty()) {
            glColor3f(0.0f, 0.0f, 1.0f);
            glPointSize(menu_point_size * 1.5f);
            glBegin(GL_POINTS);
            for (const auto& pos : kdtree_copy) {
                glVertex3f(pos.x(), pos.y(), pos.z());
            }
            glEnd();
            
            // 绘制蓝色连线（较细，避免完全遮挡黄色线）
            if (frame_id_copy >= 0) {
                glColor3f(0.0f, 0.5f, 1.0f);
                glLineWidth(menu_line_width * 0.8f);  // 稍细一些
                for (const auto& pos : kdtree_copy) {
                    glBegin(GL_LINES);
                    glVertex3f(current_copy.x(), current_copy.y(), current_copy.z());
                    glVertex3f(pos.x(), pos.y(), pos.z());
                    glEnd();
                }
            }
        }
        
        // 绘制坐标轴
        glLineWidth(3.0f);
        glBegin(GL_LINES);
        // X轴 - 红色
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
        // Y轴 - 绿色
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
        // Z轴 - 蓝色
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();
        
        pangolin::FinishFrame();
    }
}

// 启动Pangolin可视化线程
void StartVisualization() {
    if (!g_viewer) {
        g_viewer = new TrajectoryViewer();
        std::thread viewer_thread(&TrajectoryViewer::Run, g_viewer);
        viewer_thread.detach();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 等待初始化
    }
}

// 更新可视化数据
void UpdateVisualization(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses,
                        int queryFrameId, const KeyFrameDB& loopCandidates,
                        const std::vector<size_t>& kdtreeCandidates, const KeyFrameDB& keyFrameDB) {
    if (!g_viewer) return;
    
    // 更新轨迹
    g_viewer->UpdateTrajectory(poses);
    
    // 提取关键帧位置
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> kf_positions;
    for (const auto& kf : keyFrameDB) {
        if (kf->mnFrameId < poses.size()) {
            kf_positions.push_back(poses[kf->mnFrameId].block<3,1>(0,3));
        }
    }
    g_viewer->UpdateKeyFrames(kf_positions);
    
    if (queryFrameId < poses.size()) {
        // 当前位置
        Eigen::Vector3d current_pos = poses[queryFrameId].block<3,1>(0,3);
        
        // KDTree候选位置
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> kdtree_positions;
        for (size_t idx : kdtreeCandidates) {
            if (idx < keyFrameDB.size() && keyFrameDB[idx]->mnFrameId < poses.size()) {
                kdtree_positions.push_back(poses[keyFrameDB[idx]->mnFrameId].block<3,1>(0,3));
            }
        }
        
        // Eigen候选位置和分数
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> eigen_positions;
        std::vector<float> scores;
        for (const auto& candidate : loopCandidates) {
            if (candidate->mnFrameId < poses.size()) {
                eigen_positions.push_back(poses[candidate->mnFrameId].block<3,1>(0,3));
                // TODO: BUGFIXME: 这里需要根据实际情况选择前后视图
                scores.push_back(candidate->mPlaceRecognitionScore_front);
            }
        }
        
        // 调试输出：检查转换后的位置数据
        if (!eigen_positions.empty()) {
            // std::cout << "UpdateVisualization: Converting " << eigen_positions.size() 
            //           << " Eigen candidates to positions" << std::endl;
        }
        
        g_viewer->UpdateLoopDetection(queryFrameId, current_pos, kdtree_positions, eigen_positions, scores);
    }
}
