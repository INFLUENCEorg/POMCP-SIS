#include "InfluencePredictor.hpp"
#include "Utils.hpp"
#include "glog/logging.h"
#include <ctime>

// the constructor: initialize a bunch of variables
InfluencePredictor::InfluencePredictor(
  const TwoStageDynamicBayesianNetwork &twoStageDBNRef, 
  const std::vector<std::string> &inputVariables, 
  const std::vector<std::string> &targetVariables):
  _twoStageDBNRef(twoStageDBNRef), 
  _inputVariables(inputVariables), 
  _targetVariables(targetVariables) {
  LOG(INFO) << "Input variables: " << PrintUtils::vectorToTupleString(_inputVariables);
  LOG(INFO) << "Target variables: " << PrintUtils::vectorToTupleString(_targetVariables);
}

RandomInfluencePredictor::RandomInfluencePredictor(
  const TwoStageDynamicBayesianNetwork &twoStageDBNRef, 
  const std::vector<std::string> &inputVariables, 
  const std::vector<std::string> &targetVariables): 
    InfluencePredictor(
      twoStageDBNRef, 
      inputVariables, 
      targetVariables) 
{
  LOG(INFO) << "Random influence predictor has been constructed.";
}

void RandomInfluencePredictor::oneStepSample(
  torch::Tensor &hiddenState, 
  const torch::Tensor &inputs, 
  bool &initial, 
  std::map<std::string, int> &resultsDict) const 
{
  for (const std::string &factorName: _targetVariables) {
    resultsDict[factorName] = (_twoStageDBNRef.getVariable(factorName)).sampleUniformly();
  }
  initial = false;
}

torch::Tensor RandomInfluencePredictor::getInitialHiddenState() const { 
  return torch::zeros({0}, _defaultTensorOptions);
}

void RandomInfluencePredictor::updateAndGetLoss(
  torch::Tensor &hiddenState, 
  const torch::Tensor &inputs, 
  bool &initial,
  const std::vector<int> &results,
  float &crossEntropy,
  float &NNEntropy) 
{
  if (initial == true) {
    crossEntropy = 0.0;
    NNEntropy = 0.0;
  } else {
    float sumEntropy = 0.0;
    float sumCrossEntropy = 0.0;
    for (const std::string &factorName: _targetVariables) {
      int numberOfCandidates = _twoStageDBNRef.getVariable(factorName).getNumberOfValues();
      float theEntropy = std::log(numberOfCandidates);
      float theCrossEntropy = std::log(numberOfCandidates);
      sumEntropy += theEntropy;
      sumCrossEntropy += theCrossEntropy;
    }

    // KL Divergence = Cross Entropy - Entropy
    crossEntropy = sumCrossEntropy;
    NNEntropy = sumEntropy;
  }

  initial = false;  
}

RecurrentInfluencePredictor::RecurrentInfluencePredictor(
  const TwoStageDynamicBayesianNetwork &twoStageDBNRef, 
  const std::vector<std::string> &inputVariables, 
  const std::vector<std::string> &targetVariables, 
  int numberOfHiddenStates):
  InfluencePredictor(twoStageDBNRef, inputVariables, targetVariables),
  _numberOfHiddenStates(numberOfHiddenStates) { 
  // get the total output size of the recurrent model
  _totalOutputSize = 0;
  for (const std::string &key: _targetVariables) {
    int numberOfValues = _twoStageDBNRef.getVariable(key).getNumberOfValues();
    _indices[key] = std::make_pair(_totalOutputSize, _totalOutputSize+numberOfValues);
    _totalOutputSize += numberOfValues;
  }
  // at::init_num_threads();
}

void RecurrentInfluencePredictor::oneStepSample(torch::Tensor &hiddenState, const torch::Tensor &inputs, bool &initial, std::map<std::string, int> &resultsDict) const {
  torch::NoGradGuard no_grad;
  if (initial == true) {
    for (const std::string &influenceSoruceVariableName: _targetVariables) {
      resultsDict[influenceSoruceVariableName] = _twoStageDBNRef.getVariable(influenceSoruceVariableName).sampleInitialValue();
    }
    initial = false;
  } else {
    // feedforward passing
    torch::Tensor exp_outputs = this->_feedforwardPassing(hiddenState, inputs).exp();

    for (int i=0; i<=_indices.size()-1; i++) {
      auto iterator = _indices.begin();
      std::advance(iterator, i);
      const std::string &key = iterator->first;
      const std::pair<int, int> &val = iterator->second;;
      int sample = (exp_outputs.index( {torch::indexing::Slice(std::get<0>(val), std::get<1>(val))})).multinomial(1, true).item<int>();
      resultsDict.at(key) = sample;
    }
  }
  initial = false;
}

void RecurrentInfluencePredictor::updateAndGetLoss(
  torch::Tensor &hiddenState, 
  const torch::Tensor &inputs, 
  bool &initial,
  const std::vector<int> &results, 
  float &crossEntropy, 
  float &NNEntropy) {

  if (initial == true) {
    crossEntropy = 0.0;
    NNEntropy = 0.0;
  } else {
    torch::NoGradGuard no_grad;
    // feedforward passing
    auto exp_outputs = this->_feedforwardPassing(hiddenState, inputs).exp();
    float sumEntropy = 0.0;
    float sumCrossEntropy = 0.0;
    int count = 0;
    for (const auto &key: _targetVariables) {
      auto &val = _indices.at(key);
      auto exp_logits = exp_outputs.index( {torch::indexing::Slice(std::get<0>(val), std::get<1>(val))});
      auto probs = exp_logits / torch::sum(exp_logits);
      auto probabilities = std::vector<float>(probs.data_ptr<float>(), probs.data_ptr<float>()+probs.numel());
      float theEntropy = 0.0;
      for (auto prob: probabilities) {
        if (prob > 0) {
          theEntropy -= prob * log(prob);
        }
      }
      float theCrossEntropy = -log(probabilities.at(results.at(count)));
      assert(theEntropy >=0);
      assert(theCrossEntropy >= 0);
      sumEntropy += theEntropy;
      sumCrossEntropy += theCrossEntropy;
      count += 1;
    }
    crossEntropy = sumCrossEntropy;
    NNEntropy = sumEntropy;
  }

  initial = false;
}

torch::Tensor RecurrentInfluencePredictor::getInitialHiddenState() const {
  return torch::zeros({_numberOfHiddenStates}, _defaultTensorOptions);
}

void RecurrentInfluencePredictor::warmUp() {
  for (int i=0; i<=19; i++) {
    torch::Tensor hiddenState = this->getInitialHiddenState();
    bool initial = true;
    std::map<std::string, int> resultsDict;
    torch::Tensor inputs = torch::randint(0, 1, _sizeOfInputs);
    this->oneStepSample(hiddenState, inputs, initial, resultsDict);
    if (initial == true) {
      LOG(FATAL) << "wrong";
    }
    this->oneStepSample(hiddenState, inputs, initial, resultsDict);
  }
}

PreTrainedGRUInfluencePredictor::PreTrainedGRUInfluencePredictor(
  const TwoStageDynamicBayesianNetwork &twoStageDBNRef, 
  const std::vector<std::string> &inputVariables, 
  const std::vector<std::string> &targetVariables, 
  int numberOfHiddenStates,
  const std::string &modelPath):
  RecurrentInfluencePredictor(twoStageDBNRef, inputVariables, targetVariables, numberOfHiddenStates) {
  _loadModelParameters(torch::jit::load(modelPath));
  LOG(INFO) << "Pretrained GRU influence predictor has been constructed.";
}

void PreTrainedGRUInfluencePredictor::_loadModelParameters(const torch::jit::Module &model){
  for (const auto &pair: model.named_parameters()) {
    if (pair.name == "gru.weight_ih_l0") {
      wxr = pair.value.index({torch::indexing::Slice(0,1*_numberOfHiddenStates)}).clone().transpose_(0,1);
      wxz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone().transpose_(0,1);
      wxn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone().transpose_(0,1);
    } else if (pair.name == "gru.weight_hh_l0") {
      whr = pair.value.index({torch::indexing::Slice(0,1*_numberOfHiddenStates)}).clone().transpose_(0,1);
      whz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone().transpose_(0,1);
      whn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone().transpose_(0,1);
    } else if (pair.name == "gru.bias_ih_l0") {
      bxr = pair.value.index({torch::indexing::Slice(0,1*_numberOfHiddenStates)}).clone();
      bxz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone();
      bxn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone();
    } else if (pair.name == "gru.bias_hh_l0") {
      bhr = pair.value.index({torch::indexing::Slice(0*_numberOfHiddenStates,1*_numberOfHiddenStates)}).clone();
      bhz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone();
      bhn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone();
    } else if (pair.name == "linear_layer.bias") {
      by = pair.value.clone();
    } else if (pair.name == "linear_layer.weight") {
      why = pair.value.clone().transpose_(0,1);
    }
    LOG(INFO) << "loaded: " << pair.name;
  }
}

inline torch::Tensor PreTrainedGRUInfluencePredictor::_feedforwardPassing(torch::Tensor &h, const torch::Tensor &tensorInputs) const {
  auto r = torch::sigmoid(torch::matmul(tensorInputs, wxr) + bxr + torch::matmul(h, whr) + bhr);
  auto z = torch::sigmoid(torch::matmul(tensorInputs, wxz) + bxz + torch::matmul(h, whz) + bhz);
  auto n = torch::tanh(torch::matmul(tensorInputs, wxn) + bxn + torch::mul(r, torch::matmul(h, whn) + bhn));
  h = (torch::mul((1-z), n) + torch::mul(z, h)).view(-1);
  auto y = (torch::matmul(h, why) + by).view(-1);
  return y;
}

PreTrainedRNNInfluencePredictor::PreTrainedRNNInfluencePredictor(
  const TwoStageDynamicBayesianNetwork &twoStageDBNRef, 
  const std::vector<std::string> &inputVariables, 
  const std::vector<std::string> &targetVariables, 
  int numberOfHiddenStates,
  const std::string &modelPath):
  RecurrentInfluencePredictor(twoStageDBNRef, inputVariables, targetVariables, numberOfHiddenStates) {
  _loadModelParameters(torch::jit::load(modelPath));
  LOG(INFO) << "Pretrained RNN influence predictor has been constructed.";
}

void PreTrainedRNNInfluencePredictor::_loadModelParameters(const torch::jit::Module &model)  {
  for (const auto &pair: model.named_parameters()) {
    if (pair.name == "gru.weight_ih_l0") {
      wxh = pair.value.clone().transpose_(0,1);
    } else if (pair.name == "gru.weight_hh_l0") {
      whh = pair.value.clone().transpose_(0,1);
    } else if (pair.name == "gru.bias_ih_l0") {
      bxh = pair.value.clone();
    } else if (pair.name == "gru.bias_hh_l0") {
      bhh = pair.value.clone();
    } else if (pair.name == "linear_layer.bias") {
      by = pair.value.clone();
    } else if (pair.name == "linear_layer.weight") {
      why = pair.value.clone().transpose_(0,1);
    }
    LOG(INFO) << "loaded: " << pair.name;
  }
}

inline torch::Tensor PreTrainedRNNInfluencePredictor::_feedforwardPassing(torch::Tensor &h, const torch::Tensor &tensorInputs) const {
  h = (torch::tanh(torch::matmul(tensorInputs, wxh) + bxh + torch::matmul(h, whh) + bhh)).view(-1);
  torch::Tensor y = (torch::matmul(h, why) + by).view(-1);
  return y;
}

TrainableRecurrentModel::TrainableRecurrentModel(
  int sizeOfInputs, 
  int sizeOfOutputs, 
  int sizeOfHiddenStates): 
    _sizeOfInputs(sizeOfInputs), 
    _sizeOfOutputs(sizeOfOutputs), 
    _sizeOfHiddenStates(sizeOfHiddenStates),
    _linear(torch::nn::LinearOptions(sizeOfHiddenStates, _sizeOfOutputs)) 
{
  this->register_module("linear", _linear);
}

TrainableGRU::TrainableGRU(
  int sizeOfInputs, 
  int sizeOfOutputs, 
  int sizeOfHiddenStates): 
    TrainableRecurrentModel(
      sizeOfInputs, 
      sizeOfOutputs, 
      sizeOfHiddenStates), 
    _gru(torch::nn::GRUOptions(
      _sizeOfInputs, 
      _sizeOfHiddenStates)) 
{
  this->register_module("GRU", _gru);
  LOG(INFO) << "Trainable GRU initialzied. " << std::to_string(_sizeOfInputs) << " x " << std::to_string(_sizeOfOutputs) << " x " << std::to_string(_sizeOfHiddenStates);
}

torch::Tensor TrainableGRU::forward(const torch::Tensor &x, torch::Tensor &hx) {
  hx = std::get<0>(_gru->forward(x.view({1, x.size(0), x.size(1)}), hx.view({1, hx.size(0), hx.size(1)})));
  return _linear(hx);
}

torch::Tensor TrainableGRU::forward(const torch::Tensor &xs) {
  // get sequence length
  int batchSize = xs.size(0);
  int seqLength = xs.size(1);
  // create empty hidden state
  torch::Tensor hx = torch::zeros({1, xs.size(0), _sizeOfHiddenStates});
  torch::Tensor hxs = std::get<0>(_gru->forward(xs.transpose(0, 1), hx));
  // expected shape: (batch_size, seq_length, hidden_size)
  // expected shape: (batch_size, seq_length, total_output_size)
  return (_linear(hxs.view({seqLength*batchSize, _sizeOfHiddenStates}))).view({seqLength, batchSize, _sizeOfOutputs}).transpose(0,1);
}

TrainableRNN::TrainableRNN(
  int sizeOfInputs, 
  int sizeOfOutputs, 
  int sizeOfHiddenStates): 
    TrainableRecurrentModel(
      sizeOfInputs, 
      sizeOfOutputs, 
      sizeOfHiddenStates), 
        _rnn(torch::nn::RNNOptions(_sizeOfInputs, _sizeOfHiddenStates)) 
{
  this->register_module("RNN", _rnn);
  LOG(INFO) << "Trainable rnn initialzied. " << std::to_string(_sizeOfInputs) << " x " << std::to_string(_sizeOfOutputs) << " x " << std::to_string(_sizeOfHiddenStates);
}

torch::Tensor TrainableRNN::forward(const torch::Tensor &x, torch::Tensor &hx) {
  hx = std::get<0>(_rnn->forward(x.view({1, x.size(0), x.size(1)}), hx.view({1, hx.size(0), hx.size(1)})));
  return _linear(hx);
}

torch::Tensor TrainableRNN::forward(const torch::Tensor &xs) {
  // get sequence length
  int batchSize = xs.size(0);
  int seqLength = xs.size(1);
  // create empty hidden state
  torch::Tensor hx = torch::zeros({1, xs.size(0), _sizeOfHiddenStates});
  torch::Tensor hxs = std::get<0>(_rnn->forward(xs.transpose(0, 1), hx));
  // expected shape: (batch_size, seq_length, hidden_size)
  // expected shape: (batch_size, seq_length, total_output_size)
  return (_linear(hxs.view({seqLength*batchSize, _sizeOfHiddenStates}))).view({seqLength, batchSize, _sizeOfOutputs}).transpose(0,1);
}

TrainableRecurrentInfluencePredictor::TrainableRecurrentInfluencePredictor(
      const TwoStageDynamicBayesianNetwork &twoStageDBNRef, 
      const std::vector<std::string> &inputVariables, 
      const std::vector<std::string> &targetVariables, 
      const YAML::Node &influencePredictorParameters): 
    RecurrentInfluencePredictor(
      twoStageDBNRef, 
      inputVariables, 
      targetVariables, 
      influencePredictorParameters["numberOfHiddenStates"].as<int>()),
    _learningRate(influencePredictorParameters["Training"]["learningRate"].as<float>()),
    _batchSize(influencePredictorParameters["Training"]["batchSize"].as<int>()),
    _trainFreq(influencePredictorParameters["Training"]["trainFreq"].as<int>()),
    _weightDecay(influencePredictorParameters["Training"]["weightDecay"].as<float>()),
    _lossType(influencePredictorParameters["Training"]["lossType"].as<std::string>()) 
{ 

  trainable = true;

  if (influencePredictorParameters["Type"].as<std::string>() == "RNN") {
    _modelPtr = new TrainableRNN(
      _sizeOfInputs, 
      _totalOutputSize, 
      _numberOfHiddenStates);
  } else if (influencePredictorParameters["Type"].as<std::string>() == "GRU") {
    _modelPtr = new TrainableGRU(
      _sizeOfInputs, 
      _totalOutputSize, 
      _numberOfHiddenStates);
  }
  _optimizerPtr = new torch::optim::Adam(
    _modelPtr->parameters(), 
    torch::optim::AdamOptions(_learningRate).weight_decay(_weightDecay));
  LOG(INFO) << "Hyperparameters: ";
  LOG(INFO) << influencePredictorParameters;
  LOG(INFO) << "Trainable recurrent influence predictor has been constructed.";
}

TrainableRecurrentInfluencePredictor::~TrainableRecurrentInfluencePredictor() {
  delete _modelPtr;
  delete _optimizerPtr;
}

double TrainableRecurrentInfluencePredictor::train(ReplayBuffer *replayBufferPtr) 
{
  if (replayBufferPtr->getTotalNumberOfDataPoints()  < _batchSize) {
    return 0.0;
  }
  _modelPtr->train();
  int seqLength = replayBufferPtr->getSeqLength();
  // feedforward passing
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sampledBatch = replayBufferPtr->sampleBatch(_batchSize, _lossType);
  torch::Tensor tensorInputs = std::get<0>(sampledBatch);
  torch::Tensor tensorTargets = std::get<1>(sampledBatch);
  torch::Tensor tensorMasks = std::get<2>(sampledBatch);
  torch::Tensor tensorOutputs = _modelPtr->forward(tensorInputs);

  std::vector<torch::Tensor> variableLosses;
  int totalNumberOfSourceVariables = (int) _targetVariables.size();
  int varCount = 0;
  int maxLength = tensorMasks.size(1);
  // construct losses source variable by source variable
  for (const auto &key: _targetVariables) {
    auto& val = _indices.at(key);        
    torch::Tensor thisVariableLoss; // averaged over all batches and potentially also sequences
    auto tensorSlices = tensorOutputs.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(std::get<0>(val), std::get<1>(val))});
    auto tensorPredictedDistributions = torch::nn::functional::softmax(tensorSlices, torch::nn::functional::SoftmaxFuncOptions(2));
    // get prediction
    auto tensorFlattenedPredictedDistributions = tensorPredictedDistributions.view({_batchSize * maxLength, std::get<1>(val)-std::get<0>(val)});
    auto tensorFlattenedTargetedDistributions = tensorTargets.index({torch::indexing::Slice(), torch::indexing::Slice(), varCount}).view(_batchSize * maxLength);
    auto variableLoss = torch::nn::functional::nll_loss(torch::log(tensorFlattenedPredictedDistributions), tensorFlattenedTargetedDistributions, torch::nn::functional::NLLLossFuncOptions().reduction(torch::kNone)); 
    auto maskedVariableLoss = torch::mul(variableLoss, tensorMasks.view(-1));
    auto averagedMaskedVariableLoss = maskedVariableLoss.sum() / tensorMasks.sum();
    thisVariableLoss = averagedMaskedVariableLoss;
    variableLosses.push_back(thisVariableLoss);
    varCount += 1;
    VLOG(5) << "averaged loss of variable " << key << ": " << std::to_string(variableLosses.back().item<double>());
  }
  // average over all variables
  auto finalAveragedLoss = torch::stack(torch::TensorList(variableLosses), 0).sum();
  // optimization
  _optimizerPtr->zero_grad();
  finalAveragedLoss.backward();
  _optimizerPtr->step();
  double doubleLoss = finalAveragedLoss.item<double>();
  VLOG(5) << "final averaged loss: " << std::to_string(doubleLoss);
  return doubleLoss;
}

double TrainableRecurrentInfluencePredictor::trainMultiple(ReplayBuffer *replayBufferPtr) 
{
  // save some data
  if (replayBufferPtr->getTotalNumberOfDataPoints() < _batchSize) {
    return 0.0;
  }
  std::vector<double> losses(_trainFreq);
  for (int i=0; i<=_trainFreq-1; i++) {
    losses.at(i) = train(replayBufferPtr);
  }
  return StatisticsUtils::getAverage(losses);
}

inline torch::Tensor TrainableRecurrentInfluencePredictor::_feedforwardPassing(
  torch::Tensor &h, 
  const torch::Tensor &tensorInputs) const
{
  _modelPtr->eval();
  h = h.view({-1, _numberOfHiddenStates});
  auto outputs = _modelPtr->forward(tensorInputs.view({-1, _sizeOfInputs}), h).view(-1);
  h = h.view(-1);
  return outputs;
}