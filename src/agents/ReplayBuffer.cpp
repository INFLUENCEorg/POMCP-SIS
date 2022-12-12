#include "ReplayBuffer.hpp"

#include <time.h>
#include "Utils.hpp"
#include <iostream>
#include <memory>
#include <fstream>

void ReplayBuffer::preallocateMemory(int upperBoundOnTheNumberOfSimulationsPerStep) 
{ 
  auto begin = std::clock();
  for (int i=0; i<=_bufferSize; i++) {
    _links.push_back(std::vector<int>());
    _inputs.push_back(std::vector<std::vector<int>>()); 
    _targets.push_back(std::vector<std::vector<int>>());
    int horizon = _seqLength + 1;
    int upperBoundOnTheNumberOfDataPointsPerRealEpisode = 
      horizon * horizon * upperBoundOnTheNumberOfSimulationsPerStep;
    for (int j=0; j<=upperBoundOnTheNumberOfDataPointsPerRealEpisode-1; j++) {
      _inputs.back().push_back(std::vector<int>(_inputSize));
      _targets.back().push_back(std::vector<int>(_targetSize));
      _links.back().push_back(0);
    }
  }
  LOG(INFO) << "Preallocating memory for the replay buffer took " << (double)(std::clock()-begin)/CLOCKS_PER_SEC;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ReplayBuffer::sampleBatch(int batchSize, const std::string &lossType)
{
  // construct the probabilities for the first layer of sampling
  std::vector<double> probs(_sizes.size(), 1.0);
  for (int i=0; i<=_sizes.size()-1; i++) {
    probs[i] *= 1.0 * _sizes[i] / _totalSize;
  }
  std::discrete_distribution episode_dist(probs.begin(), probs.end());
  std::vector<torch::Tensor> tensorListOfInputs;
  std::vector<torch::Tensor> tensorListOfTargets;
  std::vector<int> episodeLengths;

  for (int batch_i=0; batch_i<=batchSize-1; batch_i++) {
    int i_episode = episode_dist(RandomUtils::getRandomEngine());
    int i_inner = std::uniform_int_distribution(0, _sizes[i_episode]-1)(RandomUtils::getRandomEngine());
    int current_index = i_inner;
    assert(current_index != -1);
    std::vector<torch::Tensor> episodeTensorListOfInputs;
    std::vector<torch::Tensor> episodeTensorListOfTargets;

    while (current_index != -1) {
      episodeTensorListOfInputs.push_back(torch::from_blob(_inputs.at(i_episode).at(current_index).data(), {1, _inputSize}, torch::TensorOptions().dtype(torch::kInt32)));
      episodeTensorListOfTargets.push_back(torch::from_blob(_targets.at(i_episode).at(current_index).data(), {1, _targetSize}, torch::TensorOptions().dtype(torch::kInt32)));
      current_index = _links.at(i_episode).at(current_index);
    }
    std::reverse(episodeTensorListOfInputs.begin(), episodeTensorListOfInputs.end());
    std::reverse(episodeTensorListOfTargets.begin(), episodeTensorListOfTargets.end());
    episodeLengths.push_back(episodeTensorListOfInputs.size());

    tensorListOfInputs.push_back(torch::cat(torch::TensorList(episodeTensorListOfInputs), 0) );
    tensorListOfTargets.push_back(torch::cat(torch::TensorList(episodeTensorListOfTargets), 0));

  }

  // PADDING
  torch::Tensor tensorInputs = torch::nn::utils::rnn::pad_sequence(tensorListOfInputs, true).set_requires_grad(false);
  torch::Tensor tensorTargets = torch::nn::utils::rnn::pad_sequence(tensorListOfTargets, true).set_requires_grad(false);

  // generating the masks
  torch::Tensor tensorMasks = torch::zeros({batchSize, tensorInputs.size(1)}, torch::TensorOptions().dtype(torch::kInt32) );
  for (int i=0; i<=batchSize-1; i++) {
    int count = 0;
    while (count <= episodeLengths[i]-2) {
      if (lossType == "mean") {
        tensorMasks[i][count] = 1;
      } else if (lossType == "last") {
        tensorMasks[i][count] = 0;
      }
      count += 1;
    }
    tensorMasks[i][count] = 1;
    count += 1;
    while (count <= tensorInputs.size(1)-1) {
      tensorMasks[i][count] = 0;
      count += 1;
    }
  }

  tensorInputs = tensorInputs.toType(torch::kFloat32);
  tensorTargets = tensorTargets.toType(torch::kLong);

  return std::make_tuple(tensorInputs, tensorTargets, tensorMasks);
}

// move the pointer to the next episode - performed once every real episode
void ReplayBuffer::wrapEpisode() 
{
  if (_episodeIndex == -1) {
    _episodeIndex += 1;
  } else {
    _sizes.push_back(_internalIndex);
    _totalSize += _internalIndex;
    _episodeIndex += 1;
    LOG(INFO) << "[REPLAY BUFFER] current number of episodes stored in the replay buffer: " << std::to_string(_episodeIndex);
    LOG(INFO) << "[REPLAY BUFFER] current number of data points stored in the replay buffer: " << std::to_string(_totalSize);
  }
  _internalIndex = 0;
}

// insert a new data point
int ReplayBuffer::insert(const std::vector<int> &stepInputs, const std::vector<int> &stepTargets, int parentIndex)
{
  _inputs.at(_episodeIndex).at(_internalIndex) = stepInputs;
  _targets.at(_episodeIndex).at(_internalIndex) = stepTargets;
  _links.at(_episodeIndex).at(_internalIndex) = parentIndex;
  _internalIndex += 1;
  // return the index of the current datapoint
  return _internalIndex-1;
}

void ReplayBuffer::save(const std::string &pathToResultsFolder){
  LOG(INFO) << "saving replay buffer.";
  // let's turn everything into three tensors

  std::vector<torch::Tensor> tensorListOfInputs;
  std::vector<torch::Tensor> tensorListOfTargets;
  std::vector<int> episodeLengths;

  for (int i_episode=0; i_episode<=_sizes.size()-1; i_episode++) {
    for (int i_inner=0; i_inner<=_sizes[i_episode]-1; i_inner++) {
      int current_index = i_inner;
      assert(current_index != -1);
      std::vector<torch::Tensor> episodeTensorListOfInputs;
      std::vector<torch::Tensor> episodeTensorListOfTargets;

      while (current_index != -1) {
        episodeTensorListOfInputs.push_back(torch::from_blob(_inputs.at(i_episode).at(current_index).data(), {1, _inputSize}, torch::TensorOptions().dtype(torch::kInt32)));
        episodeTensorListOfTargets.push_back(torch::from_blob(_targets.at(i_episode).at(current_index).data(), {1, _targetSize}, torch::TensorOptions().dtype(torch::kInt32)));
        current_index = _links.at(i_episode).at(current_index);
      }
      std::reverse(episodeTensorListOfInputs.begin(), episodeTensorListOfInputs.end());
      std::reverse(episodeTensorListOfTargets.begin(), episodeTensorListOfTargets.end());
      episodeLengths.push_back(episodeTensorListOfInputs.size());

      tensorListOfInputs.push_back(torch::cat(torch::TensorList(episodeTensorListOfInputs), 0) );
      tensorListOfTargets.push_back(torch::cat(torch::TensorList(episodeTensorListOfTargets), 0));
    }
  }

  // PADDING
  torch::Tensor tensorInputs = torch::nn::utils::rnn::pad_sequence(tensorListOfInputs, true).set_requires_grad(false);
  torch::Tensor tensorTargets = torch::nn::utils::rnn::pad_sequence(tensorListOfTargets, true).set_requires_grad(false);

  // generating the masks
  torch::Tensor tensorMasks = torch::zeros({_totalSize, tensorInputs.size(1)}, torch::TensorOptions().dtype(torch::kInt32) );
  for (int i=0; i<=_totalSize-1; i++) {
    int count = 0;
    while (count <= episodeLengths[i]-1) {
      tensorMasks[i][count] = 1;
      count += 1;
    }
    // tensorMasks[i][count] = 1;
    // count += 1;
    // while (count <= tensorInputs.size(1)-1) {
    //   tensorMasks[i][count] = 0;
    //   count += 1;
    // }
  }

  tensorInputs = tensorInputs.toType(torch::kFloat32);
  tensorTargets = tensorTargets.toType(torch::kLong);

  // then save locally
  std::vector<char> fInputs = torch::pickle_save(tensorInputs);
  std::vector<char> fTargets = torch::pickle_save(tensorTargets);
  std::vector<char> fMasks = torch::pickle_save(tensorMasks);
  std::ofstream fInputsOut(pathToResultsFolder+"/inputs.zip", std::ios::out | std::ios::binary);
  fInputsOut.write(fInputs.data(), fInputs.size());
  fInputsOut.close();
  std::ofstream fTargetsOut(pathToResultsFolder+"/outputs.zip", std::ios::out | std::ios::binary);
  fTargetsOut.write(fTargets.data(), fTargets.size());
  fTargetsOut.close();
  std::ofstream fMasksOut(pathToResultsFolder+"/masks.zip", std::ios::out | std::ios::binary);
  fMasksOut.write(fMasks.data(), fMasks.size());
  fMasksOut.close();
  LOG(INFO) << "replay buffer saved.";
}