#include "PlanningAgentSimulator.hpp"
#include "domains/Domain.hpp"
#include <algorithm>
#include <chrono>

PlanningAgentSimulator::PlanningAgentSimulator(
  const std::string &IDOfAgentToControl, 
  const Domain* const domainPtr): 
  _IDOfAgentToControl(IDOfAgentToControl), 
  _domainPtr(domainPtr), 
  _DBNRef(domainPtr->getDBN()) {}

GlobalSimulator::GlobalSimulator(
  const std::string &IDOfAgentToControl, 
  const Domain* const domainPtr, 
  const YAML::Node &fullAgentParameters): 
  PlanningAgentSimulator(IDOfAgentToControl, domainPtr)  {
  // build up the agent simulators
  int counter = 0;
  for (YAML::const_iterator it = fullAgentParameters.begin(); it != fullAgentParameters.end(); it++) {
    std::string agentID = it->first.as<std::string>();
    if (agentID != _IDOfAgentToControl) {
      std::string agentType = it->second["Type"].as<std::string>();
      agentSimulators[agentID] = std::unique_ptr<AtomicAgentSimulator>(_domainPtr->makeAtomicAgentSimulator(agentID, agentType, fullAgentParameters[agentID]));
      agentStateIndices[agentID] = counter;
      counter +=  1 + 2 * (_domainPtr->getNumberOfStepsToPlan());
    }
  }
  _sizeOfAOH = counter;
  VLOG(1) << _domainPtr->getDomainName() + " single agent global simulator has been built.";
}

void GlobalSimulator::updateState(GlobalSimulatorState &state) const  {
  // send observations to the corresponding agents
  for (auto &[agentID, startIndex]: agentStateIndices) {
    int agentObs = _DBNRef.getValueOfVariableFromIndex("o"+agentID, state.environmentState);
    agentSimulators.at(agentID)->observe(state.AOH.begin()+agentStateIndices.at(agentID), agentObs);
  }
}

void GlobalSimulator::step(GlobalSimulatorState &state, int action, int &observation, float &reward, bool &done) const  {
  auto begin = std::clock();
  VLOG(6) << "[GLOBAL SIMULATOR] [SIMULATION] Started to do one step simulation.";
  // simulate actions of other agents
  for (auto &[agentID, agentSimulator]: agentSimulators) {
    int simulatedAction =  agentSimulator->step(state.AOH.begin()+agentStateIndices.at(agentID));
    VLOG(6) << "[GLOBAL SIMULATOR] [SIMULATION] Agent " << agentID << " is simulated to take action " << std::to_string(simulatedAction) << ".";
    state.environmentState["a"+agentID] = simulatedAction;
  }
  state.environmentState["a"+_IDOfAgentToControl] = action;
  VLOG(6) << "[GLOBAL SIMULATOR] [SIMULATION] Finished sampling actions of other agents.";
  _DBNRef.step(state.environmentState, "full");
  VLOG(6) << "[GLOBAL SIMULATOR] [SIMULATION] Finished one step simulation in the DBN.";
  observation = _DBNRef.getValueOfVariableFromIndex("o"+_IDOfAgentToControl, state.environmentState);
  reward = _DBNRef.getValueOfVariableFromIndex("r"+_IDOfAgentToControl, state.environmentState);
  this->updateState(state);
  done = false;
  if (state.environmentState["t"] == _domainPtr->getNumberOfStepsToPlan()) {
    done = true;
  }
  VLOG(6) << "[GLOBAL SIMULATOR] [SIMULATION] Finished one step simulation in the global simulator.";
  VLOG(5) << "[GS] step simulation took " << std::to_string((double)(std::clock()-begin)/CLOCKS_PER_SEC);
}

GlobalSimulator::GlobalSimulatorState GlobalSimulator::sampleInitialState() const {
  std::vector<int> AOH(_sizeOfAOH);
  for (auto &[agentID, startIndex]: agentStateIndices) {
    AOH[startIndex] = 1; // 1 means writing starts from index 1
  }
  return GlobalSimulatorState(_domainPtr->sampleInitialState(), AOH);
}

InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulator(
  const std::string &IDOfAgentToControl, 
  const Domain* const domainPtr, 
  const YAML::Node &simulatorParameters): 
  PlanningAgentSimulator(IDOfAgentToControl, domainPtr)  
{
  // construct the local model  
  _domainPtr->constructLocalModel(IDOfAgentToControl, _localStates, _influenceSourceVariables, _destinationFactors, _DSeparationVariablesPerStep);
}

InfluenceAugmentedLocalSimulator::~InfluenceAugmentedLocalSimulator() {
  delete _influencePredictorPtr;
}

void InfluenceAugmentedLocalSimulator::updateState(InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState &state, int action) const {
  int count = 0;
  auto accessor = state.influencePredictorInputs.accessor<float, 1>();
  for (const auto &DSeparationVariable: _DSeparationVariablesPerStep) {
    if (DSeparationVariable[0] == 'a') {
      accessor[count] = action;
    } else {
      accessor[count] = state.environmentState.at(DSeparationVariable);
    }
    count += 1;
  }
  state.initial = false;
}

void InfluenceAugmentedLocalSimulator::warmUp() {
  this->_influencePredictorPtr->warmUp();
}

void InfluenceAugmentedLocalSimulator::step(
  InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState &state, 
  int action, 
  int &observation, 
  float &reward, 
  bool &done) const {
  auto begin = std::clock();
  VLOG(6) << "[SIMULATION] [IALS] action: " << action;
  state.environmentState["a"+this->_IDOfAgentToControl] = action;
  VLOG(6) << "[SIMULATION] [IALS] before inf prediction: " << PrintUtils::mapToTupleString(state.environmentState);
  //auto outgoingInfluenceStrength = computeOutgoingInfluenceStrength(state.environmentState, action)
  auto begin2 = std::clock();
  _influencePredictorPtr->oneStepSample(state.influencePredictorState, state.influencePredictorInputs, state.initial, state.environmentState);
  VLOG(6) << "one step sample took " << std::to_string((double)(std::clock()-begin2)/CLOCKS_PER_SEC);
  VLOG(6) << "[SIMULATION] [IALS] after inf prediction: " << PrintUtils::mapToTupleString(state.environmentState);
  auto begin4 = std::clock();
  _DBNRef.step(state.environmentState, "local");
  VLOG(6) << "local simulation took " << std::to_string((double)(std::clock()-begin4)/CLOCKS_PER_SEC);
  VLOG(6) << "[SIMULATION] [IALS] after environment simulation: " << PrintUtils::mapToTupleString(state.environmentState);
  auto begin3 = std::clock();
  reward = _DBNRef.getValueOfVariableFromIndex("r"+this->_IDOfAgentToControl, state.environmentState);
  observation = _DBNRef.getValueOfVariableFromIndex("o"+this->_IDOfAgentToControl, state.environmentState);
  VLOG(6) << "query took " << std::to_string((double)(std::clock()-begin3)/CLOCKS_PER_SEC);
  auto begin6 = std::clock();
  this->updateState(state, action);
  VLOG(6) << "update state took " << std::to_string((double)(std::clock()-begin6)/CLOCKS_PER_SEC);
  done =false;
  if (state.environmentState["t"] == _domainPtr->getNumberOfStepsToPlan()) {
    done = true;
  }
  VLOG(5) << "[IALS] step simulation took " << std::to_string((double)(std::clock()-begin)/CLOCKS_PER_SEC);
}

InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState InfluenceAugmentedLocalSimulator::sampleInitialState() const {
  std::map<std::string, int> environmentState;
  // sample a full environment state
  std::map<std::string, int> fullSampledState = _domainPtr->sampleInitialState();
  // take the local states out
  for (const auto &varName: _localStates) {
    environmentState[varName] = fullSampledState[varName];
  }
  torch::Tensor influencePredictorInputs = torch::zeros((int)_DSeparationVariablesPerStep.size(), _defaultTensorOptions);
  torch::Tensor influencePredictorState = _influencePredictorPtr->getInitialHiddenState();
  return InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState(environmentState, influencePredictorInputs, influencePredictorState);
}

SelfImprovingSimulator::SelfImprovingSimulator(
  const std::string &IDOfAgentToControl, 
  const Domain* const domain, 
  const YAML::Node &simulatorParameters, 
  const YAML::Node &fullAgentParameters):
  PlanningAgentSimulator::PlanningAgentSimulator(IDOfAgentToControl, domain),
  GlobalSimulator::GlobalSimulator(IDOfAgentToControl, domain, fullAgentParameters),
  InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulator(IDOfAgentToControl, domain, simulatorParameters)
{
  // initialize the influence predictor
  this->initializeInfluencePredictor(simulatorParameters);
  // initialize the replay buffer
  this->initializeReplayBuffer(simulatorParameters);
}

void SelfImprovingSimulator::initializeInfluencePredictor(const YAML::Node &simulatorParameters) {
  std::string type = simulatorParameters["InfluencePredictor"]["Type"].as<std::string>();
  if (simulatorParameters["InfluencePredictor"]["modelPath"].IsDefined() == false && (type == "GRU" || type == "RNN")) {
    _influencePredictorPtr = new TrainableRecurrentInfluencePredictor(
      this->_DBNRef, 
      this->_DSeparationVariablesPerStep,  
      this->_influenceSourceVariables, 
      simulatorParameters["InfluencePredictor"]
    );
  } else {
    InfluenceAugmentedLocalSimulator::initializeInfluencePredictor(simulatorParameters);
  }
}

void SelfImprovingSimulator::initializeReplayBuffer(const YAML::Node &simulatorParameters) {

  int bufferSize = this->_domainPtr->getParameters()["Experiment"]["repeat"].as<int>();
  if (simulatorParameters["ReplayBuffer"].IsDefined() && simulatorParameters["ReplayBuffer"]["bufferSize"].IsDefined()) {
    bufferSize = simulatorParameters["ReplayBuffer"]["bufferSize"].as<int>();
  }
  std::string bufferType = "Tree";
  int inputSize = this->_DSeparationVariablesPerStep.size();
  int targetSize = this->_influenceSourceVariables.size();
  int horizon = _domainPtr->getNumberOfStepsToPlan();

  // build up the replay buffer
  _replayBufferPtr = new ReplayBuffer(inputSize, targetSize, horizon, bufferSize);
  // preallocate enough memory for the replay buffer
  int upperBoundOnTheNumberOfSimulationsPerStep;
  const YAML::Node &agentRolloutParameters = _domainPtr->getParameters()["AgentComponent"][_IDOfAgentToControl]["Rollout"];
  if (agentRolloutParameters["numberOfSimulationsPerStep"].IsDefined() == true) {
    upperBoundOnTheNumberOfSimulationsPerStep = agentRolloutParameters["numberOfSimulationsPerStep"].as<int>();
  } else if (simulatorParameters["ReplayBuffer"]["upperBoundOnTheNumberOfSimulationsPerStep"].IsDefined() == true) {
    upperBoundOnTheNumberOfSimulationsPerStep = simulatorParameters["ReplayBuffer"]["upperBoundOnTheNumberOfSimulationsPerStep"].as<int>();
  } else {
    upperBoundOnTheNumberOfSimulationsPerStep = 10000;
  }
  _replayBufferPtr->preallocateMemory(upperBoundOnTheNumberOfSimulationsPerStep);
}

SelfImprovingSimulator::~SelfImprovingSimulator() {
  delete _influencePredictorPtr;
  delete _replayBufferPtr;
}

void SelfImprovingSimulator::reset() {
  _replayBufferPtr->wrapEpisode();
}

void SelfImprovingSimulator::train(YAML::Node &results) {
  // before training, keep the record of how much data we have now
  int numberOfEpisodes = _replayBufferPtr->getEpisodeIndex();
  int numberOfDataPoints = _replayBufferPtr->getTotalNumberOfDataPoints();
  results["numberOfDataPoints"] = numberOfDataPoints;
  results["numberOfEpisodes"] = numberOfEpisodes;
  if (_influencePredictorPtr->trainable == true) {
    double averageLoss = static_cast<TrainableRecurrentInfluencePredictor*>(_influencePredictorPtr)->trainMultiple(_replayBufferPtr);
    if (averageLoss > 0) {
      results["average_train_loss"] = averageLoss;
      LOG(INFO) << "[TRAINING] Average train loss: " << std::to_string(averageLoss);
    }
  }
}

void SelfImprovingSimulator::step(
  IndexedGlobalSimulatorState &state, 
  int action, 
  int &observation, 
  float &reward, 
  bool &done, 
  float &crossEntropy, 
  float &entropy,
  int entropyEstimationType,
  bool saveData) 
{
  auto begin = std::clock();
  std::vector<int> inputs(this->_DSeparationVariablesPerStep.size());
  std::vector<int> targets(this->_influenceSourceVariables.size(), -1);
  // simulate actions of other agents
  float actionEntropy = 0.0;
  float previousStateEntropy = state.stateInfluenceSourceEntropy;
  for (auto &[agentID, agentSimulator]: this->agentSimulators) {
    int simulatedAction;
    std::string actionID = "a" + agentID;
    if (std::find(_influenceSourceVariables.begin(), _influenceSourceVariables.end(), actionID) != _influenceSourceVariables.end()) {
      float perActionEntropy;
      simulatedAction = agentSimulator->step(state.globalSimulatorState.AOH.begin()+this->agentStateIndices.at(agentID), perActionEntropy);
      actionEntropy += perActionEntropy;
    } else {
      simulatedAction = agentSimulator->step(state.globalSimulatorState.AOH.begin()+this->agentStateIndices.at(agentID));
    }
    state.globalSimulatorState.environmentState[actionID] = simulatedAction;
  }
  if (state.globalSimulatorState.environmentState["t"] > 0) {
    if (state.initial == true) {
      LOG(FATAL) << "initial should only be true in the beginning";
    }
    // prepare inputs - the current local states and the previous action, which should be still part of the global state
    for (int i=0; i<=this->_DSeparationVariablesPerStep.size()-1; i++) {
      inputs.at(i) = state.globalSimulatorState.environmentState.at(_DSeparationVariablesPerStep.at(i));
      if (_DSeparationVariablesPerStep.at(i)[0] == 'a') {
        assert(inputs.at(i) == state.globalSimulatorState.environmentState.at("a"+_IDOfAgentToControl));
      }
    }
    // prepare targets - the influence source states and the actions of agents (which are unknown at this point)
    int i=0;
    for (const std::string &influenceSourceVariable: this->_influenceSourceVariables) {
      targets.at(i) = state.globalSimulatorState.environmentState.at(influenceSourceVariable);
      i += 1;
    }
  }
  // compute the loss and update the hidden state of the influence predictor
  float NNCrossEntropy;
  float NNEntropy;
  this->_influencePredictorPtr->updateAndGetLoss(
    state.influencePredictorState, 
    state.influencePredictorInputs, 
    state.initial, 
    targets, 
    NNCrossEntropy, 
    NNEntropy);
  assert(NNEntropy >= 0);
  assert(NNCrossEntropy >= 0);
  assert(state.initial == false);

  if (state.globalSimulatorState.environmentState["t"] > 0) {
    // compute entropy
    if (entropyEstimationType == NONE) {
      entropy = 0.0;
    } else if (entropyEstimationType == MLE) {
      LOG(FATAL) << "MLE type of entropy estimation is not supported.";
    } else if (entropyEstimationType == NN) {
      entropy = NNEntropy;
    } else if (entropyEstimationType == EXACT) {
      entropy = previousStateEntropy + actionEntropy;
    }
    // compute cross entropy
    crossEntropy = NNCrossEntropy;
  } else {
    entropy = 0.0;
    crossEntropy = 0.0;
  }

  if (entropy < 0 || crossEntropy < 0) {
    LOG(FATAL) << "entropy and cross entropy should all be non-negative";
  }
  // perform a simulation step in the DBN
  state.globalSimulatorState.environmentState["a" + _IDOfAgentToControl] = action;
  std::map<std::string, float> stateEntropies;
  _DBNRef.stepWithEntropy(state.globalSimulatorState.environmentState, "full", this->_influenceSourceVariables, stateEntropies);
  assert(state.globalSimulatorState.environmentState["t"] >= 1);

  // compute next state entropies
  state.stateInfluenceSourceEntropy = 0.0;
  for (const auto &[key, stateEntropy]: stateEntropies) {
    state.stateInfluenceSourceEntropy += stateEntropy;
  }

  assert(state.stateInfluenceSourceEntropy >= 0);

  observation = _DBNRef.getValueOfVariableFromIndex("o"+_IDOfAgentToControl, state.globalSimulatorState.environmentState);
  reward = _DBNRef.getValueOfVariableFromIndex("r"+_IDOfAgentToControl, state.globalSimulatorState.environmentState);

  if (state.globalSimulatorState.environmentState["t"] == _domainPtr->getNumberOfStepsToPlan()) {
    done = true;
  } else {
    done = false;
  }

  // update global state
  GlobalSimulator::updateState(state.globalSimulatorState);

  // update influence predictor inputs
  auto accessor = state.influencePredictorInputs.accessor<float, 1>();
  for (int i=0; i<=this->_DSeparationVariablesPerStep.size()-1; i++) {
    accessor[i] = state.globalSimulatorState.environmentState.at(_DSeparationVariablesPerStep.at(i));
    if (_DSeparationVariablesPerStep.at(i)[0] == 'a') {
      assert(accessor[i] == action);
    }
  }

  // update initial
  state.initial = false;

  // save data when it is not the first step and save data is true
  // now adding something, save data only when using not fixed influence predictor
  if (this->_influencePredictorPtr->trainable == true) {
    if (saveData == true && state.globalSimulatorState.environmentState["t"] >= 2) {
      int index = state.index;
      int nextIndex = _replayBufferPtr->insert(inputs, targets, index);
      state.index = nextIndex;
    }
  }

  VLOG(5) << "[SIS] step simulation took " << std::to_string((double)(std::clock()-begin)/CLOCKS_PER_SEC);
}

void SelfImprovingSimulator::step(
  IndexedGlobalSimulatorState &state, 
  int action, 
  int &observation, 
  float &reward, 
  bool &done) {
  float crossEntropy; 
  float entropy;
  this->step(
    state,
    action,
    observation,
    reward,
    done,
    crossEntropy, 
    entropy, 
    NONE, 
    false);
}

void SelfImprovingSimulator::warmUp() {
  this->_influencePredictorPtr->warmUp();
}