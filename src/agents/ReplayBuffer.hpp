#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include <random>
#include <tuple>
#include "torch/torch.h"
#include <assert.h>
#include <deque>

class ReplayBuffer {
  public:
    ReplayBuffer(
      int inputSize, 
      int targetSize, 
      int horizon, 
      int bufferSize=-1):
       _inputSize(inputSize), 
       _targetSize(targetSize), 
       _seqLength(horizon-1), 
       _bufferSize(bufferSize) 
    {
      if (bufferSize <= 0 ) LOG(FATAL) << "buffer size <= 0";
      LOG(INFO) << "Input size: " 
                << std::to_string(inputSize) 
                << ", Output size: " 
                << std::to_string(targetSize) 
                << ", sequence length: " 
                << std::to_string(_seqLength);
    }

    // preallocate memory for the data
    void preallocateMemory(int upperBoundOnTheNumberOfSimulationsPerStep);

    // sample a batch of data
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sampleBatch(int batchSize = 32, const std::string &lossType="last");

    inline int getTotalNumberOfDataPoints() const { return _totalSize; }
    
    void wrapEpisode();
    
    inline int getSeqLength() const { return _seqLength; }
    
    int insert(const std::vector<int> &stepInputs, const std::vector<int> &stepOutputs, int parentIndex);

    inline int getEpisodeIndex() const { return _episodeIndex; }

    void save(const std::string &pathToResultsFolder);

  private:
    int _inputSize;
    int _targetSize;
    int _bufferSize;
    int _seqLength;

    // the data
    std::deque<std::vector<int>> _links; // need to be saved
    std::deque<std::vector<std::vector<int>>> _inputs; // need to be saved
    std::deque<std::vector<std::vector<int>>> _targets; // need to be saved
    std::deque<int> _sizes;

    int _episodeIndex = -1;
    int _internalIndex = 0;
    
    int _totalSize = 0;
};

#endif