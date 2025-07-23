/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "KeyFrameDatabase.h"

#include "KeyFrame.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM3
{

void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    mvDatabase.insert(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    if (!mvDatabase.count(pKF)) return;
    
    mvDatabase.erase(pKF);
}

void KeyFrameDatabase::clear()
{
    unique_lock<mutex> lock(mMutex);
    mvDatabase.clear();
}

void KeyFrameDatabase::clearMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(auto it = mvDatabase.begin(), it_next = mvDatabase.begin(); it != mvDatabase.end();  it = it_next)
    {
        it_next++;
        KeyFrame* pKFi = *it;
        if (pMap == pKFi->GetMap())
        {
            // Dont delete the KF because the class Map clean all the KF when it is destroyed
            mvDatabase.erase(it);
        }
    }
}

bool compFirst(const pair<float, KeyFrame*> & a, const pair<float, KeyFrame*> & b)
{
    return a.first > b.first;
}

void KeyFrameDatabase::DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, 
                                             vector<KeyFrame*> &vpMergeCand, int nNumCandidates)
{
    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    // 候选帧集合
    set<KeyFrame*> spCandidiateKF;
    {
        unique_lock<mutex> lock(mMutex);

        float bestScore = 0;
        // 将当前帧的全局描述子映射为Eigen形式的矩阵
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
                queryDescriptors (pKF->mGlobalDescriptors.ptr<float>(), 
                pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);

        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            // Compute the distance of global descriptors
            // Eigen is much faster than OpenCV Mat
            assert(!pKFi->mGlobalDescriptors.empty());

            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
                    dbDescriptors (pKFi->mGlobalDescriptors.ptr<float>(), 
                    pKFi->mGlobalDescriptors.rows, 
                    pKFi->mGlobalDescriptors.cols);


            // 两个方向相反的单位向量【0，1】【0，-1】之间作差的2范数是sqrt(2) 
            /*
            具体来说，两个单位向量的差向量的2范数越大，表示它们的方向之间的夹角越大，
            即它们指向的方向越远离彼此。当两个单位向量方向相同时，它们的差向量的2范数为0；
            当它们方向相反时，差向量的2范数达到最大值 
            */ 
            // (queryDescriptors - dbDescriptors).norm() 越小表示越相近，mPlaceRecognitionScore得分也越大
            pKFi->mPlaceRecognitionScore = std::max(0.f, 1 - (queryDescriptors - dbDescriptors).norm());
            // 标记关键帧队列中的此帧被当前关键帧访问过
            pKFi->mnPlaceRecognitionQuery = pKF->mnId;
            bestScore = max(pKFi->mPlaceRecognitionScore, bestScore);
        }
        // 最小分数为最佳分数的0.8倍数
        float minScore = bestScore * 0.8f;

        cout << "In mvDatabase bestScore is : " << bestScore <<endl;
        cout << "In mvDatabase minScore is : " << minScore <<endl;

        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            if (pKFi->mPlaceRecognitionScore > minScore && pKFi->mnPlaceRecognitionQuery == pKF->mnId)
                // 如果关键帧集合中候选帧的得分大于0.8倍的候选帧，则加入候选帧集合
                spCandidiateKF.insert(pKFi);
        }
    }

    cout << "After minScore filiter, spCandidiateKF nums is : " << spCandidiateKF.size() <<endl;

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // 通过共试关系积累分数
    for(auto it=spCandidiateKF.begin(), itend=spCandidiateKF.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = *it;
        // 获取10帧最近共试图
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = pKFi->mPlaceRecognitionScore;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                continue;
            // 累计打分
            accScore+=pKF2->mPlaceRecognitionScore;
            if(pKF2->mPlaceRecognitionScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }

        }
        // 当前帧的累计得分
        pKFi->mPlaceRecognitionAccScore = accScore;
        // 保存累计得分与本组中最佳相似帧
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        // 更新最佳累计得分
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }
    // 排序 从高到低
    lAccScoreAndMatch.sort(compFirst);
    //  nNumCandidates ：3
    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;
    int i = 0;
    list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin();
    while(i < lAccScoreAndMatch.size() && (vpLoopCand.size() < nNumCandidates || vpMergeCand.size() < nNumCandidates))
    {
        KeyFrame* pKFi = it->second;
        if(pKFi->isBad())
            continue;
        // 如果set中没有pKFi
        if(!spAlreadyAddedKF.count(pKFi))
        {
            if(pKF->GetMap() == pKFi->GetMap() && vpLoopCand.size() < nNumCandidates)
            {
                vpLoopCand.push_back(pKFi);
                cout << "The finnally candidate fream score is :" << it->first<< endl;
            }
            else if(pKF->GetMap() != pKFi->GetMap() && vpMergeCand.size() < nNumCandidates && !pKFi->GetMap()->IsBad())
            {
                vpMergeCand.push_back(pKFi);
            }
            spAlreadyAddedKF.insert(pKFi);
        }
        i++;
        it++;
    }
}


vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F, Map* pMap)
{
    set<KeyFrame*> spCandidiateKF;
    {
        unique_lock<mutex> lock(mMutex);

        float bestScore = 0;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(
            F->mGlobalDescriptors.ptr<float>(), 
            F->mGlobalDescriptors.rows, F->mGlobalDescriptors.cols);
            
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            // Compute the distance of global descriptors
            // Eigen is much faster than OpenCV Mat
            assert(!pKFi->mGlobalDescriptors.empty());
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> dbDescriptors(pKFi->mGlobalDescriptors.ptr<float>(), pKFi->mGlobalDescriptors.rows, pKFi->mGlobalDescriptors.cols);
            pKFi->mRelocScore = std::max(0.f, 1 - (queryDescriptors - dbDescriptors).norm());
            pKFi->mnRelocQuery = F->mnId;
            bestScore = max(pKFi->mRelocScore, bestScore);
        }

        const float thresholdScore = 0.5;
        float minScore = std::max(thresholdScore, bestScore * 0.8f);
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            if (pKFi->mRelocScore > minScore && pKFi->mnRelocQuery == F->mnId)
                spCandidiateKF.insert(pKFi);
        }
    }

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(auto it=spCandidiateKF.begin(), itend=spCandidiateKF.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = *it;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = pKFi->mRelocScore;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        pKFi->mRelocAccScore = accScore;
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    lAccScoreAndMatch.sort(compFirst);

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if (pKFi->GetMap() != pMap)
                continue;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
