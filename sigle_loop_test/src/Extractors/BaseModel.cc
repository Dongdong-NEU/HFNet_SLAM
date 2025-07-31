#include "Extractors/BaseModel.h"
#include "Extractors/HFNetRTModel.h"

#include <unordered_set>

using namespace std;

namespace ORB_SLAM3
{

std::vector<BaseModel*> gvpModels;
BaseModel* gpGlobalModel = nullptr;

void InitAllModels(const std::string& strModelPath, ModelType modelType, cv::Size ImSize, int nLevels, float scaleFactor)
{
    if (gvpModels.size())
    {
        for (auto pModel : gvpModels) delete pModel;
        gvpModels.clear();
    }
    gvpModels.reserve(nLevels);
    float scale = 1.0f;
    for (int level = 0; level < nLevels; ++level)
    {
        cv::Vec4i inputShape{1, cvRound(ImSize.height * scale), cvRound(ImSize.width * scale), 1};
        BaseModel *pNewModel = nullptr;
        ModelDetectionMode mode;
        if (modelType == kHFNetRTModel)
        {
            if (level == 0) mode = kImageToLocalAndGlobal;
            else mode = kImageToLocal;
            pNewModel = InitRTModel(strModelPath, mode, inputShape);
        }
        else
        {
            cerr << "Wrong type of model!" << endl;
            exit(-1);
        }
        gvpModels.emplace_back(pNewModel);
        scale /= scaleFactor;
    }

    if (gpGlobalModel) delete gpGlobalModel;
    cv::Vec4i inputShape{1, ImSize.height / 8, ImSize.width / 8, 96};
    BaseModel *pNewModel = nullptr;
    if (modelType == kHFNetRTModel)
    {
        pNewModel = nullptr;
    }
    else
    {
        cerr << "Wrong type of model!" << endl;
        exit(-1);
    }
    gpGlobalModel = pNewModel;
}

std::vector<BaseModel*> GetModelVec(void)
{
    if (gvpModels.empty())
    {
        cerr << "Try to get models before initialize them" << endl;
        exit(-1);
    }
    return gvpModels;
}

BaseModel* GetGlobalModel(void)
{
    if (gvpModels.empty())
    {
        cerr << "Try to get global model before initialize it" << endl;
        exit(-1);
    }
    return gpGlobalModel;
}


BaseModel* InitRTModel(const std::string& strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape)
{
    BaseModel* pModel;
    pModel = new HFNetRTModel(strModelPath, mode, inputShape);
    if (pModel->IsValid())
    {
        cout << "Successfully loaded HFNet TensorRT model."
             << " Mode: " << gStrModelDetectionName[mode]
             << " Shape: " << inputShape.t() << endl;
    }
    else exit(-1);

    return pModel;
}

// copy from tensorflow.contrib.resampler
void Resampler(const float* data, const float* warp, float* output,
               const int batch_size, const int data_height, 
               const int data_width, const int data_channels, const int num_sampling_points)
{
    const int warp_batch_stride = num_sampling_points * 2;
    const int data_batch_stride = data_height * data_width * data_channels;
    const int output_batch_stride = num_sampling_points * data_channels;
    const float zero = static_cast<float>(0.0);
    const float one = static_cast<float>(1.0);

    auto resample_batches = [&](const int start, const int limit) {
      for (int batch_id = start; batch_id < limit; ++batch_id) {
        // Utility lambda to access data point and set output values.
        // The functions take care of performing the relevant pointer
        // arithmetics abstracting away the low level details in the
        // main loop over samples. Note that data is stored in NHWC format.
        auto set_output = [&](const int sample_id, const int channel,
                              const float value) {
          output[batch_id * output_batch_stride + sample_id * data_channels +
                 channel] = value;
        };

        auto get_data_point = [&](const int x, const int y, const int chan) {
          const bool point_is_in_range =
              (x >= 0 && y >= 0 && x <= data_width - 1 && y <= data_height - 1);
          return point_is_in_range
                     ? data[batch_id * data_batch_stride +
                            data_channels * (y * data_width + x) + chan]
                     : zero;
        };

        for (int sample_id = 0; sample_id < num_sampling_points; ++sample_id) {
          const float x = warp[batch_id * warp_batch_stride + sample_id * 2];
          const float y = warp[batch_id * warp_batch_stride + sample_id * 2 + 1];
          // The interpolation function:
          // a) implicitly pads the input data with 0s (hence the unusual checks
          // with {x,y} > -1)
          // b) returns 0 when sampling outside the (padded) image.
          // The effect is that the sampled signal smoothly goes to 0 outside
          // the original input domain, rather than presenting a jump
          // discontinuity at the image boundaries.
          if (x > static_cast<float>(-1.0) && y > static_cast<float>(-1.0) &&
              x < static_cast<float>(data_width) &&
              y < static_cast<float>(data_height)) {
            // Precompute floor (f) and ceil (c) values for x and y.
            const int fx = std::floor(static_cast<float>(x));
            const int fy = std::floor(static_cast<float>(y));
            const int cx = fx + 1;
            const int cy = fy + 1;
            const float dx = static_cast<float>(cx) - x;
            const float dy = static_cast<float>(cy) - y;

            for (int chan = 0; chan < data_channels; ++chan) {
              const float img_fxfy = dx * dy * get_data_point(fx, fy, chan);
              const float img_cxcy =
                  (one - dx) * (one - dy) * get_data_point(cx, cy, chan);
              const float img_fxcy = dx * (one - dy) * get_data_point(fx, cy, chan);
              const float img_cxfy = (one - dx) * dy * get_data_point(cx, fy, chan);
              set_output(sample_id, chan,
                         img_fxfy + img_cxcy + img_fxcy + img_cxfy);
            }
          } else {
            for (int chan = 0; chan < data_channels; ++chan) {
              set_output(sample_id, chan, zero);
            }
          }
        }
      }
    };
    
    resample_batches(0, batch_size);
}

std::vector<cv::KeyPoint> NMS(const std::vector<cv::KeyPoint> &vToDistributeKeys, int width, int height, int radius)
{
    std::vector<std::vector<const cv::KeyPoint*>> vpKeypoints(height, vector<const cv::KeyPoint*>(width, nullptr));
    unordered_set<const cv::KeyPoint*> selected;
    for (const auto &kpt : vToDistributeKeys)
    {
        vpKeypoints[kpt.pt.y][kpt.pt.x] = &kpt;
        selected.insert(&kpt);
    }

    for (const auto &kpt : vToDistributeKeys)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            for (int dy = -radius; dy <= radius; ++dy)
            {
                const int x = kpt.pt.x + dx;
                const int y = kpt.pt.y + dy;
                if (x < 0 || y < 0 || x >= width || y >= height) continue;
                if (!vpKeypoints[y][x]) continue;

                if (vpKeypoints[kpt.pt.y][kpt.pt.x]->response < vpKeypoints[y][x]->response)
                {
                    selected.erase(vpKeypoints[kpt.pt.y][kpt.pt.x]);
                    vpKeypoints[kpt.pt.y][kpt.pt.x] = nullptr;
                    goto jump;
                }
            }
        }
        jump: ;
    }

    std::vector<cv::KeyPoint> res;
    res.reserve(selected.size());
    for (const auto &pKP : selected)
    {
        res.emplace_back(*pKP);
    }
    return res;
}

}