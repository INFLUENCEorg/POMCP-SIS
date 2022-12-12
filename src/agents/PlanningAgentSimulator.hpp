#ifndef PLANNING_AGENT_SIMULATOR_HPP_
#define PLANNING_AGENT_SIMULATOR_HPP_

#include <influence/InfluencePredictor.hpp>
#include "dbns/TwoStageDynamicBayesianNetwork.hpp"
#include <agents/AtomicAgent.hpp>
#include <variant>
#include <random>
#include <memory>
#include "ReplayBuffer.hpp"
#include <map>

#define GS 0
#define IALS 1

#define NN 0
#define MLE 1
#define EXACT 2
#define NONE 3

class Domain;

// the base simulator
class PlanningAgentSimulator {
  public:
    PlanningAgentSimulator(const std::string &IDOfAgentToControl, const Domain* const domainPtr);
    virtual void reset() {};
    virtual void warmUp() {};
  
  protected:
    const Domain* const _domainPtr;
    const std::string &_IDOfAgentToControl;
    const TwoStageDynamicBayesianNetwork &_DBNRef;
};

// the global simulator
class GlobalSimulator: virtual public PlanningAgentSimulator {
  public:
    
    // the state of global simulator
    class GlobalSimulatorState {
      public:
        std::map<std::string, int> environmentState;
        // the AOH of other agents // can also be map of vectors
        std::vector<int> AOH; 

        GlobalSimulatorState(
          const GlobalSimulatorState &stateToCopy): 
          environmentState(stateToCopy.environmentState), 
          AOH(stateToCopy.AOH) {}
        
        GlobalSimulatorState(
          std::map<std::string, int> environmentState, 
          std::vector<int> AOH): 
          environmentState(environmentState), 
          AOH(AOH) {}
    };

    GlobalSimulator(
      const std::string &IDOfAgentToControl, 
      const Domain* const domainPtr, 
      const YAML::Node &fullAgentParameters);

    void step(
      GlobalSimulatorState &state, 
      int action, 
      int &observation, 
      float &reward, 
      bool &done) const;

    GlobalSimulatorState sampleInitialState() const;

  protected:
    
    int _sizeOfAOH = 0;

    void updateState(GlobalSimulatorState &state) const;

    std::map<std::string, int> agentStateIndices;
    std::map<std::string, std::unique_ptr<AtomicAgentSimulator>> agentSimulators;
};

// the influence-augmented local simulator
class InfluenceAugmentedLocalSimulator: virtual public PlanningAgentSimulator {
  public:

    // the state of influence-augmented local simulator
    class InfluenceAugmentedLocalSimulatorState {
      public:
        bool initial = true;
        std::map<std::string, int> environmentState;
        torch::Tensor influencePredictorInputs;
        torch::Tensor influencePredictorState;

        InfluenceAugmentedLocalSimulatorState(const InfluenceAugmentedLocalSimulatorState &stateToCopy): 
          environmentState(stateToCopy.environmentState), 
          initial(stateToCopy.initial), 
          influencePredictorState(stateToCopy.influencePredictorState.clone()), 
          influencePredictorInputs(stateToCopy.influencePredictorInputs.clone()) {}
        
        InfluenceAugmentedLocalSimulatorState(
          std::map<std::string, int> environmentState,
          torch::Tensor influencePredictorInputs,
          torch::Tensor influencePredictorState): 
          environmentState(environmentState), 
          influencePredictorInputs(influencePredictorInputs), 
          influencePredictorState(influencePredictorState) {}
    };

    InfluenceAugmentedLocalSimulator(
      const std::string &IDOfAgentToControl, 
      const Domain* const domainPtr, 
      const YAML::Node &simulatorParameters); 

    ~InfluenceAugmentedLocalSimulator();

    void warmUp() override;

    void step(
      InfluenceAugmentedLocalSimulatorState &state, 
      int action, 
      int &observation, 
      float &reward, 
      bool &done) const;

    InfluenceAugmentedLocalSimulatorState sampleInitialState() const;

    // construct the influence predictor

    // random influence predictor and fixed influence predictors
    virtual void initializeInfluencePredictor(const YAML::Node &simulatorParameters) {
      std::string influencePredictorType = simulatorParameters["InfluencePredictor"]["Type"].as<std::string>();
      if (influencePredictorType == "Random") {
        _influencePredictorPtr = new RandomInfluencePredictor(_DBNRef, _DSeparationVariablesPerStep, _influenceSourceVariables);
      } else {
        std::string modelPath = simulatorParameters["InfluencePredictor"]["modelPath"].as<std::string>();
        int numberOfHiddenStates = simulatorParameters["InfluencePredictor"]["numberOfHiddenStates"].as<int>();
        if (influencePredictorType == "RNN") {
          _influencePredictorPtr = new PreTrainedRNNInfluencePredictor(_DBNRef, _DSeparationVariablesPerStep, _influenceSourceVariables, numberOfHiddenStates, modelPath);
        } else if (influencePredictorType == "GRU") {
          _influencePredictorPtr = new PreTrainedGRUInfluencePredictor(_DBNRef, _DSeparationVariablesPerStep, _influenceSourceVariables, numberOfHiddenStates, modelPath);
        } else {
          LOG(FATAL) << "Influence predictor type " << influencePredictorType << " is not supported.";
        }
      } 
    }

  protected:

    // the influence predictor
    InfluencePredictor* _influencePredictorPtr;

    void updateState(InfluenceAugmentedLocalSimulatorState &state, int action) const;
    
    torch::TensorOptions _defaultTensorOptions = torch::TensorOptions().requires_grad(false).dtype(torch::kFloat32);

    // the influence variables
    std::vector<std::string> _influenceSourceVariables;
    std::vector<std::string> _localStates;
    std::vector<std::string> _destinationFactors;
    std::vector<std::string> _DSeparationVariablesPerStep;
};

class SelfImprovingSimulator: public GlobalSimulator, public InfluenceAugmentedLocalSimulator {
  public:

    // the constructor
    SelfImprovingSimulator(
      const std::string &IDOfAgentToControl, 
      const Domain* const domain, 
      const YAML::Node &simulatorParameters, 
      const YAML::Node &fullAgentParameters);

    void initializeInfluencePredictor(const YAML::Node &simulatorParameters) override;

    void initializeReplayBuffer(const YAML::Node &simulatorParameters);

    ~SelfImprovingSimulator();

    void reset() override;

    void train(YAML::Node &results);

    ReplayBuffer* getReplayBufferPtr() {return _replayBufferPtr; }

    class IndexedGlobalSimulatorState {
      public:
        int index = -1;
        GlobalSimulator::GlobalSimulatorState globalSimulatorState;

        bool initial = true; // whether this is an initial state
        torch::Tensor influencePredictorInputs;
        torch::Tensor influencePredictorState; // the hidden state of the influence predictor
        float stateInfluenceSourceEntropy = 0.0; // at previous time step? yes

        IndexedGlobalSimulatorState(const IndexedGlobalSimulatorState &stateToCopy): 
          globalSimulatorState(stateToCopy.globalSimulatorState), 
          index(stateToCopy.index), 
          initial(stateToCopy.initial), 
          influencePredictorInputs(stateToCopy.influencePredictorInputs.clone()), 
          influencePredictorState(stateToCopy.influencePredictorState.clone()), 
          stateInfluenceSourceEntropy(stateToCopy.stateInfluenceSourceEntropy) {}

        IndexedGlobalSimulatorState(
          GlobalSimulator::GlobalSimulatorState globalSimulatorState,
          torch::Tensor influencePredictorInputs,
          torch::Tensor influencePredictorState
        ): globalSimulatorState(globalSimulatorState), 
          influencePredictorInputs(influencePredictorInputs), 
          influencePredictorState(influencePredictorState) {
          initial = true;
          stateInfluenceSourceEntropy = 0.0;
        }
    };

    IndexedGlobalSimulatorState sampleInitialGlobalState() {
      return IndexedGlobalSimulatorState(
        GlobalSimulator::sampleInitialState(),
        torch::zeros({(int)this->_DSeparationVariablesPerStep.size()}, _defaultTensorOptions),
        _influencePredictorPtr->getInitialHiddenState()
      );
    }

    InfluenceAugmentedLocalSimulatorState sampleInitialIALSState() {
      return InfluenceAugmentedLocalSimulator::sampleInitialState();
    }

    void warmUp() override;

    void step(
      IndexedGlobalSimulatorState &state, 
      int action, 
      int &observation, 
      float &reward, 
      bool &done, 
      float &crossEntropy, 
      float &entropy,
      int entropyEstimationType=NONE,
      bool saveData=false);

    void step(
      IndexedGlobalSimulatorState &state, 
      int action, 
      int &observation, 
      float &reward, 
      bool &done);

    void step(
      InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState &state,
      int action,
      int &observation,
      float &reward,
      bool &done
    ) { InfluenceAugmentedLocalSimulator::step(state, action, observation, reward, done); };

  private:
    // the replay buffer
    mutable ReplayBuffer *_replayBufferPtr;
};

#endif