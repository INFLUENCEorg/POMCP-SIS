#ifndef INFLUENCE_PREDICTOR_HPP_
#define INFLUENCE_PREDICTOR_HPP_

#include "dbns/TwoStageDynamicBayesianNetwork.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include "agents/ReplayBuffer.hpp"
#include <vector>

// the general influence predictor
class InfluencePredictor {
  public:

    // the constructor
    InfluencePredictor(
      const TwoStageDynamicBayesianNetwork &TwoStageDBNRef, 
      const std::vector<std::string> &inputVariables, 
      const std::vector<std::string> &targetVariables);

    // core method: sample one step of influence source variables
    virtual void oneStepSample(
      torch::Tensor &hiddenState, 
      const torch::Tensor &inputs, 
      bool &initial, 
      std::map<std::string, int> &dict) const = 0;

    // get the initial hidden state of the influence predictor
    virtual torch::Tensor getInitialHiddenState() const = 0;
    
    // (abstract) compute the one-step loss
    virtual void updateAndGetLoss(
      torch::Tensor &hiddenState, 
      const torch::Tensor &inputs, 
      bool &initial,
      const std::vector<int> &results,
      float &crossEntropy,
      float &NNEntropy) = 0;

    virtual void warmUp() {};

    bool trainable = false;

  protected:
    const TwoStageDynamicBayesianNetwork &_twoStageDBNRef;

    // D separation variables per step
    std::vector<std::string> _inputVariables;

    // influence source variables
    std::vector<std::string> _targetVariables;

    int _sizeOfInputs = _inputVariables.size();
    std::unordered_map<std::string, std::pair<int, int>> _indices;

    torch::TensorOptions _defaultTensorOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
};

// the random influence predictor 
class RandomInfluencePredictor: public InfluencePredictor  {
  public:
    RandomInfluencePredictor(
      const TwoStageDynamicBayesianNetwork &TwoStageDBNRef, 
      const std::vector<std::string> &inputVariables, 
      const std::vector<std::string> &targetVariables);

    void oneStepSample(
      torch::Tensor &hiddenState, 
      const torch::Tensor &inputs, 
      bool &initial, 
      std::map<std::string, int> &resultsDict) const override;

    torch::Tensor getInitialHiddenState() const override;

    void updateAndGetLoss(
      torch::Tensor &hiddenState, 
      const torch::Tensor &inputs, 
      bool &initial,
      const std::vector<int> &results,
      float &crossEntropy,
      float &NNEntropy) override;
};

// the general influence predictor that is based on recurrent neural networks
class RecurrentInfluencePredictor: public InfluencePredictor {
  public:
    RecurrentInfluencePredictor(
      const TwoStageDynamicBayesianNetwork &TwoStageDBNRef, 
      const std::vector<std::string> &inputVariables, 
      const std::vector<std::string> &targetVariables, 
      int numberOfHiddenStates);
    
    virtual void oneStepSample(
      torch::Tensor &hiddenState, 
      const torch::Tensor &inputs, 
      bool &initial, 
      std::map<std::string, int> &resultsDict) const override;

    void updateAndGetLoss(
      torch::Tensor &hiddenState, 
      const torch::Tensor &inputs, 
      bool &initial,
      const std::vector<int> &results,
      float &crossEntropy,
      float &NNEntropy) override;

    torch::Tensor getInitialHiddenState() const override;

    void warmUp() override;

  protected:
    int _numberOfHiddenStates;
    int _totalOutputSize;
    // expect the hidden state to be flattened and updated within the function
    inline virtual torch::Tensor _feedforwardPassing(torch::Tensor &tensorHidden, const torch::Tensor &tensorInputs) const = 0;
};

// Pretrained GRU influence predictor
class PreTrainedGRUInfluencePredictor: public RecurrentInfluencePredictor {
  public:
    PreTrainedGRUInfluencePredictor(
      const TwoStageDynamicBayesianNetwork &TwoStageDBNRef, 
      const std::vector<std::string> &inputVariables, 
      const std::vector<std::string> &targetVariables, 
      int numberOfHiddenStates,
      const std::string &modelPath);
  protected:
    void _loadModelParameters(const torch::jit::Module &model);
    inline torch::Tensor _feedforwardPassing(torch::Tensor &h, const torch::Tensor &tensorInputs) const override;
  private:
    // the parameters of the GRU
    torch::Tensor wxr;
    torch::Tensor bxr;
    torch::Tensor whr;
    torch::Tensor bhr;
    torch::Tensor wxz;
    torch::Tensor bxz;
    torch::Tensor whz;
    torch::Tensor bhz;
    torch::Tensor wxn;
    torch::Tensor bxn;
    torch::Tensor whn;
    torch::Tensor bhn;
    torch::Tensor by;
    torch::Tensor why;
};

// Pretrained RNN influence predictor 
class PreTrainedRNNInfluencePredictor: public RecurrentInfluencePredictor {
  public:
    PreTrainedRNNInfluencePredictor(
      const TwoStageDynamicBayesianNetwork &TwoStageDBNRef, 
      const std::vector<std::string> &inputVariables, 
      const std::vector<std::string> &targetVariables, 
      int numberOfHiddenStates,
      const std::string &modelPath);
  protected:
    void _loadModelParameters(const torch::jit::Module &model);
    inline torch::Tensor _feedforwardPassing(torch::Tensor &h, const torch::Tensor &tensorInputs) const override;
  private:
    // the parameters of the RNN
    torch::Tensor wxh;
    torch::Tensor bxh;
    torch::Tensor whh;
    torch::Tensor bhh;
    torch::Tensor why;
    torch::Tensor by;
};

// the recurrent classifer model
class TrainableRecurrentModel: public torch::nn::Module {
  public:
    TrainableRecurrentModel(int sizeOfInputs, int sizeOfOutputs, int sizeOfHiddenStates);

    virtual torch::Tensor forward(const torch::Tensor &xs) = 0;

    virtual torch::Tensor forward(const torch::Tensor &x, torch::Tensor &hx) = 0;

  protected:
    int _sizeOfInputs;
    int _sizeOfOutputs;
    int _sizeOfHiddenStates;
    torch::nn::Linear _linear;
};

// the GRU model
class TrainableGRU: public TrainableRecurrentModel {
  public:
    TrainableGRU(int sizeOfInputs, int sizeOfOutputs, int sizeOfHiddenStates);
    
    // step-by-step forward: return the output and change the hidden state in-place
    // expected shape of x: (batch_size, input_size)
    // expected shape of hx: (batch_size, hidden_size)
    // expected shape of output: (batch_size, total_output_size)
    // IMPORTANT: the hidden state needs to be updated here
    torch::Tensor forward(const torch::Tensor &x, torch::Tensor &hx) override;

    // trajectory-by-trajectory forward
    // expected shape of x: (batch_size, seq_length(fixed), input_size)
    // expected shape of hx
    torch::Tensor forward(const torch::Tensor &xs) override;

  protected:
    torch::nn::GRU _gru;
};

// the RNN model
class TrainableRNN: public TrainableRecurrentModel {
  public:
    TrainableRNN(int sizeOfInputs, int sizeOfOutputs, int sizeOfHiddenStates);

    // step-by-step forward: return the output and change the hidden state in-place
    // expected shape of x: (batch_size, input_size)
    // expected shape of hx: (batch_size, hidden_size)
    // expected shape of output: (batch_size, total_output_size)
    // IMPORTANT: the hidden state needs to be updated here
    torch::Tensor forward(const torch::Tensor &x, torch::Tensor &hx) override;

    // trajectory-by-trajectory forward
    // expected shape of x: (batch_size, seq_length(fixed), input_size)
    // expected shape of hx
    torch::Tensor forward(const torch::Tensor &xs) override;

  protected:
    torch::nn::RNN _rnn;
};

// Trainable Recurrent Influence Predictor
class TrainableRecurrentInfluencePredictor: public RecurrentInfluencePredictor {
  public:
    TrainableRecurrentInfluencePredictor(
      const TwoStageDynamicBayesianNetwork &TwoStageDBNRef, 
      const std::vector<std::string> &inputVariables, 
      const std::vector<std::string> &targetVariables, 
      const YAML::Node &influencePredictorParameters);

    ~TrainableRecurrentInfluencePredictor();

    double train(ReplayBuffer *replayBufferPtr);

    double trainMultiple(ReplayBuffer *replayBufferPtr);

  protected:
    inline torch::Tensor _feedforwardPassing(torch::Tensor &h, const torch::Tensor &tensorInputs) const override;

  private:
    int _batchSize;
    float _learningRate;
    int _trainCounter = 0;
    int _trainFreq;
    float _weightDecay;
    mutable TrainableRecurrentModel *_modelPtr;
    mutable torch::optim::Adam *_optimizerPtr;
    std::vector<double> _trainLosses;
    std::string _lossType;
};

#endif