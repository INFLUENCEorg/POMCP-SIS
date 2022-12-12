#include "PlanningAgent.hpp"
#include "Utils.hpp"
#include <math.h>
#include <ctime>

template class POMCPAtomicAgent<GlobalSimulator::GlobalSimulatorState, GlobalSimulator>;
template class POMCPAtomicAgent<InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState, InfluenceAugmentedLocalSimulator>;

AbstractPOMCPAtomicAgent::AbstractPOMCPAtomicAgent(
  const std::string &agentID, 
  int numberOfActions, 
  int numberOfStepsToPlan,
  const YAML::Node &agentParameters
): AtomicAgent(
  agentID, 
  numberOfActions, 
  numberOfStepsToPlan, 
  agentParameters),
  _looper(this)
{
  // the planning horizon for the current step
  _planningHorizon = _numberOfStepsToPlan;
  // the discount factor
  _discountFactor = _parameters["Rollout"]["discountFactor"].as<float>();
  // number of initial particles
  _numberOfParticles = _parameters["Rollout"]["numberOfParticles"].as<int>();
  // discount horizon
  _discountHorizon = _parameters["Rollout"]["discountHorizon"].as<float>();
  // particle reinvigoration
  _particleReinvigoration = _parameters["Rollout"]["particleReinvigoration"].as<bool>();
  _particleReinvigorationRate = _parameters["Rollout"]["particleReinvigorationRate"].as<float>();
  // exploration constant
  _explorationConstant = _parameters["Rollout"]["explorationConstant"].as<float>();
  // simulation termination condition
  if (_parameters["Rollout"]["numberOfSimulationsPerStep"].IsDefined() == true) {
    _numberOfSimulationsPerStep = _parameters["Rollout"]["numberOfSimulationsPerStep"].as<int>();
  }
  if (_parameters["Rollout"]["numberOfSecondsPerStep"].IsDefined() == true) {
    _numberOfSecondsPerStep = _parameters["Rollout"]["numberOfSecondsPerStep"].as<double>();
  }
  // if we are going to reuse Monte Carlo simulations for particle filtering
  _reuseSimulationsToFilterParticle = true;
  if (_parameters["Rollout"]["reuseSimulationsToFilterParticle"].IsDefined() == true) {
    _reuseSimulationsToFilterParticle = _parameters["Rollout"]["reuseSimulationsToFilterParticle"].as<bool>();
  }
};

void AbstractPOMCPAtomicAgent::reset() 
{
  // NOTE THAT THIS IS NOT REALLY USED BY THE PLANNING AGENT
  _AOH.clear();
  _AOH.reserve(1 + 2 * _numberOfStepsToPlan);
  _AOH[0] = 1;
  // reset planning horizon
  _planningHorizon = _numberOfStepsToPlan;
}

int AbstractPOMCPAtomicAgent::act(YAML::Node results)
{
  if (results.size() == 0) {
    assert(_planningHorizon == _numberOfStepsToPlan);
    this->_initializeResultsYAML(results);
  }
  int selectedAction;
  if (this->_particleDepleted() == true) {
    VLOG(3) << "[Agent " + _agentID + "]: taking random action because of particle depletion";
    selectedAction = std::experimental::randint(0, _numberOfActions-1);
  } else {
    VLOG(3) << "[Agent " + _agentID + "]: started to do planning with horizon " + std::to_string(_planningHorizon) + ".";
    selectedAction = plan(results);
  }
  // wrap up planning
  _previousActionTaken = selectedAction;
  // decrease planning horizon by one
  _planningHorizon -= 1;
  VLOG(3) << "[Agent " + _agentID + "]: selected action " + std::to_string(selectedAction) + ".";
  VLOG(3) << "--------------------------------------------------";
  return selectedAction;
}

template <class A>
int AbstractPOMCPAtomicAgent::selectAction(ObservationNode<A>* nodePtr1, bool UCB) const {
  int selectedAction;
  int bestAction = -1;
  float bestValue;
  // iterate over all actions
  for (int actionID=0; actionID<=nodePtr1->numberOfActions-1; actionID++) {
    // if an action has never been taken before, then just take it
    if (nodePtr1->childrenNodes.at(actionID)->_N == 0) {
      return actionID;
    }
    float value = nodePtr1->childrenNodes.at(actionID)->_Q;
    if (UCB == true) {
      value += this->computeExplorationBonus(nodePtr1->_N, nodePtr1->childrenNodes.at(actionID)->_N, _explorationConstant);
    }
    if (bestAction == -1 || value >= bestValue) {
      bestAction = actionID;
      bestValue = value;
    }
  }
  selectedAction = bestAction;
  return selectedAction;
}

template <class A, class B>
int AbstractPOMCPAtomicAgent::selectAction(ObservationNode<A>* nodePtr1, ObservationNode<B>* nodePtr2, bool UCB) const {
  if (nodePtr2 == nullptr) {
    return this->selectAction(nodePtr1, UCB);
  } else if (nodePtr1 == nullptr) {
    return this->selectAction(nodePtr2, UCB);
  }
  int totalN = nodePtr1->_N + nodePtr2->_N;
  int totalN_ = 0;
  int selectedAction;
  int bestAction = -1;
  float bestValue;
  // iterate over all actions
  for (int actionID=0; actionID<=nodePtr1->numberOfActions-1; actionID++) {
    int actionN1 = nodePtr1->childrenNodes.at(actionID)->_N;
    int actionN2 = nodePtr2->childrenNodes.at(actionID)->_N;
    int actionN = actionN1 + actionN2;
    // if an action has never been taken before, then just take it
    if (actionN == 0) {
      return actionID;
    }
    float actionQ1 = nodePtr1->childrenNodes.at(actionID)->_Q;
    float actionQ2 = nodePtr2->childrenNodes.at(actionID)->_Q;
    float actionQ = (actionQ1 * actionN1 + actionQ2 * actionN2) / (actionN1 + actionN2);
    totalN_ += actionN;
    float value = actionQ;
    if (UCB == true) {
      float explorationBonus = this->computeExplorationBonus(totalN, actionN, _explorationConstant);
      value += explorationBonus;
    }
    if (bestAction == -1 || value >= bestValue) {
      bestAction = actionID;
      bestValue = value;
    }
  }
  assert(totalN_ == totalN);
  selectedAction = bestAction;
  return selectedAction;
}

template <class SimulatorState, class Simulator> 
void AbstractPOMCPAtomicAgent::_updateSearchTree(SearchTree<SimulatorState> *searchTreePtr, Simulator *simulatorPtr, int action, int observation) 
{  
  // there is no need t o perform update of search tree if particle depletion has already occurred
  if (searchTreePtr->particleDepleted() == true) { return; }

  // obtain the old and new root node
  ObservationNode<SimulatorState> *newRootObservationNodePtr = searchTreePtr->pop(action, observation);
  ObservationNode<SimulatorState> *currentRootObservationNodePtr = searchTreePtr->_rootObservationNodePtr;

  // perform the additional particle filtering process
  if (_reuseSimulationsToFilterParticle == false) {
    newRootObservationNodePtr->particles.reserve(_numberOfParticles);
    int count = 0;
    int simObservation;
    float simReward;
    bool simDone;
    while (true) {
      SimulatorState sampledState = searchTreePtr->_rootObservationNodePtr->sampleParticle();
      simulatorPtr->step(sampledState, action, simObservation, simReward, simDone);
      if (simObservation == observation) {
        newRootObservationNodePtr->particles.push_back(sampledState);
        if (newRootObservationNodePtr->particles.size() == _numberOfParticles) {
          break;
        }
      } else {
        if ((count >= 100 * _numberOfParticles) && newRootObservationNodePtr->particles.size() == 0) {
          break;
        } else if (count >= 1000 * _numberOfParticles) {
          break;
        }
      }
      count += 1;
    }
    VLOG(3) << "[Agent " + _agentID + "]: number of particles left after the filtering step: " << std::to_string(newRootObservationNodePtr->particles.size()); 
  } 

  if (_particleReinvigoration == true) {
    LOG(FATAL) << "Particle Reinvigoration not implemented yet!";
    VLOG(3) << "[Agent " + _agentID + "]: number of particles left before the observing step: " << std::to_string(newRootObservationNodePtr->particles.size());
  }
  // prune the tree even if the particle depletion has already occurred?
  searchTreePtr->reset(newRootObservationNodePtr);
}

template <class SimulatorState, class Simulator>
float AbstractPOMCPAtomicAgent::rollout(
  SimulatorState &state, 
  Simulator *simulatorPtr, 
  int horizon, 
  int depth, 
  std::vector<float> *crossEntropies, 
  std::vector<float> *entropies)
{
  // initialize variables
  float discounted_return = 0.0;
  float factor = 1.0;
  float cumulativeFactor = std::pow(_discountFactor, depth);

  // temporary variables
  int action;
  int observation;
  float reward;
  bool done = false;

  // the loop
  for (int step=0; step<=horizon-1; step++) {
    if (cumulativeFactor < _discountHorizon) {
      break;
    }

    if (done == true) {
      LOG(FATAL) << "should have been done";
    }

    // pick a random action
    action = std::experimental::randint(0, _numberOfActions-1);

    // simulation
    if constexpr (std::is_same_v<SimulatorState, SelfImprovingSimulator::IndexedGlobalSimulatorState>) {
      float crossEntropy;
      float entropy;
      MixedPOMCPAtomicAgent* thisPtr = static_cast<MixedPOMCPAtomicAgent*>(this);
      simulatorPtr->step(state, action, observation, reward, done, crossEntropy, entropy, thisPtr->_entropyEstimationType, thisPtr->_storeRolloutData);
      if (thisPtr->_computeRolloutLoss == true) {
        crossEntropies->push_back(crossEntropy);
        entropies->push_back(entropy);
      }
    } 
    if constexpr (std::is_same_v<SimulatorState, SelfImprovingSimulator::IndexedGlobalSimulatorState> == false) {
      simulatorPtr->step(state, action, observation, reward, done);
    }

    // update variables
    discounted_return += factor * reward;
    factor *= _discountFactor;
    cumulativeFactor *= _discountFactor; 
  }
  return discounted_return;
}

float AbstractPOMCPAtomicAgent::computeExplorationBonus(int NTotal, int N, float explorationConstant) const {
  assert(NTotal >= N);
  return explorationConstant * sqrtf(std::log(NTotal)/N);
}

template <class SimulatorState1, class SimulatorState2, class Simulator>
float AbstractPOMCPAtomicAgent::simulate(
  SimulatorState1 &sampledState,
  Simulator *simulatorPtr,
  ObservationNode<SimulatorState1> *thisNodePtr,
  int horizon,
  int depth,  
  ObservationNode<SimulatorState2> *otherNodePtr,
  std::vector<float> *crossEntropies,
  std::vector<float> *entropies) 
{
  assert(thisNodePtr != nullptr);
  assert(horizon == _planningHorizon - depth);
  // if the horizon is 0 or the depth is larger than the effective horizon then terminate without doing anything?
  if (horizon == 0 || std::pow(_discountFactor, depth) < _discountHorizon) {
    return 0.0;
  } else {
    bool isRoot = thisNodePtr->isRoot();
    if (isRoot == false && _reuseSimulationsToFilterParticle == true) {
      assert(depth != 0);
      assert(horizon < _planningHorizon);
      thisNodePtr->particles.push_back(SimulatorState1(sampledState));
    }
    // pick an UCB action
    int action = this->selectAction(thisNodePtr, otherNodePtr, true);
    // one step simulation
    int observation;
    float reward;
    bool done;
    if constexpr (std::is_same_v<SimulatorState1, SelfImprovingSimulator::IndexedGlobalSimulatorState> ) {
      float crossEntropy;
      float entropy;
      simulatorPtr->step(sampledState, action, observation, reward, done, crossEntropy, entropy, static_cast<MixedPOMCPAtomicAgent*>(this)->_entropyEstimationType, true);
      crossEntropies->push_back(crossEntropy);
      entropies->push_back(entropy);
    } else {
      simulatorPtr->step(sampledState, action, observation, reward, done);
    }
    float Return = reward;
    if (done == false) {
      ObservationNode<SimulatorState1> *nextThisNodePtr = thisNodePtr->getNextObservationNode(action, observation);
      ObservationNode<SimulatorState2> *nextOtherNodePtr = nullptr;
      if (otherNodePtr != nullptr) {
        nextOtherNodePtr = otherNodePtr->getNextObservationNode(action, observation);
      }
      bool doRollout;
      if (nextThisNodePtr != nullptr) {
        doRollout = false;
      } else {
        thisNodePtr->addNextObservationNode(action, observation);
        if (nextThisNodePtr == nullptr && nextOtherNodePtr != nullptr) {
          nextThisNodePtr = thisNodePtr->getNextObservationNode(action, observation);
          assert(nextThisNodePtr != nullptr);
          doRollout = false;
        } else if (nextThisNodePtr == nullptr && nextOtherNodePtr == nullptr) {
          doRollout = true;
        }
      }
      if (doRollout == false) {
        if constexpr (std::is_same_v<SimulatorState1, SelfImprovingSimulator::IndexedGlobalSimulatorState>) {
          Return += _discountFactor * this->simulate(
            sampledState,
            simulatorPtr,
            nextThisNodePtr,
            horizon-1,
            depth+1,
            nextOtherNodePtr,
            crossEntropies,
            entropies);
        } else {
          Return += _discountFactor * this->simulate(
            sampledState,
            simulatorPtr,
            nextThisNodePtr,
            horizon-1,
            depth+1,
            nextOtherNodePtr);
        } 
      } else {
        VLOG(4) << "Rollout started after " << std::to_string(depth+1) << " step, with " << std::to_string(horizon-1) << " steps to go.";
        if constexpr (std::is_same_v<SimulatorState1, SelfImprovingSimulator::IndexedGlobalSimulatorState>) {
          Return += _discountFactor * this->rollout(
            sampledState, 
            simulatorPtr,
            horizon-1, 
            depth+1, 
            crossEntropies,
            entropies);
        } else {
          Return += _discountFactor * this->rollout(
            sampledState, 
            simulatorPtr,
            horizon-1, 
            depth+1);
        }
      }
    } else {
      assert(depth == _planningHorizon-1);
      assert(horizon == 1);
    }
    thisNodePtr->update(Return);
    thisNodePtr->childrenNodes.at(action)->update(Return);
    return Return;
  }
}

template <class SimulatorState, class Simulator> 
POMCPAtomicAgent<SimulatorState, Simulator>::POMCPAtomicAgent(
  const std::string &agentID, 
  int numberOfActions, 
  int numberOfStepsToPlan, 
  const YAML::Node &agentParameters, 
  Simulator* simulatorPtr): 
  _simulatorPtr(simulatorPtr),
  AbstractPOMCPAtomicAgent(
    agentID, 
    numberOfActions, 
    numberOfStepsToPlan, 
    agentParameters) 
{
  _searchTreePtr = new SearchTree<SimulatorState>(_numberOfActions);
  LOG(INFO) << "A POMCP Agent has been created.";
}

template <class SimulatorState, class Simulator>
POMCPAtomicAgent<SimulatorState, Simulator>::~POMCPAtomicAgent() 
{
  // the agent owns the simulator and the search tree (the database)
  delete _searchTreePtr;
  delete _simulatorPtr;
}

// the observation function
template <class SimulatorState, class Simulator>
void POMCPAtomicAgent<SimulatorState, Simulator>::observe(int observation) 
{
  VLOG(3) << "[Agent " + _agentID + "]: observed " + std::to_string(observation) + ".";
  if (_planningHorizon == 0) {
    return;
  }
  bool particleDepletedAlready = this->_particleDepleted();
  this->_updateSearchTree(_searchTreePtr, _simulatorPtr, _previousActionTaken, observation);
  if (particleDepletedAlready == false && this->_particleDepleted() == true) {
    LOG(INFO) << "Particle depletion has occurred with " << std::to_string(_planningHorizon) << " to go.";
  }
  VLOG(3) << "--------------------------------------------------";
}

template <class SimulatorState, class Simulator>
void POMCPAtomicAgent<SimulatorState, Simulator>::reset() 
{
  AbstractPOMCPAtomicAgent::reset();
  // destroy the previous search tree and build a new one
  _searchTreePtr->reset();
  // sample initial states
  for (int i=0; i<=_numberOfParticles-1; i++) {
    _searchTreePtr->_rootObservationNodePtr->particles.push_back(_simulatorPtr->sampleInitialState());
  }
  // reset the simulator
  _simulatorPtr->reset();
}

template <class SimulatorState, class Simulator>
void POMCPAtomicAgent<SimulatorState, Simulator>::_initializeResultsYAML(YAML::Node results) {
  results["number_of_particles_before_simulation"] = YAML::Node();
  results["number_of_seconds_per_step"] = YAML::Node();
  results["number_of_simulations_per_step"] = YAML::Node();
}

template <class SimulatorState, class Simulator>
int POMCPAtomicAgent<SimulatorState, Simulator>::plan(YAML::Node &results)
{
  VLOG(3) << "[Agent " << _agentID << "]: number of particles before simulation: " << std::to_string(this->_searchTreePtr->_rootObservationNodePtr->particles.size());
  results["number_of_particles_before_simulation"].push_back(this->_searchTreePtr->_rootObservationNodePtr->particles.size());
  int selectedAction;
  _looper.reset();
  while (_looper.finished() == false) {

    _simulatorPtr->warmUp();

    _looper.start();

    // sample a root state
    SimulatorState sampledState = _searchTreePtr->_rootObservationNodePtr->sampleParticle();

    // perform the simulate function
    // the simulate function will update the tree
    this->simulate<SimulatorState, SimulatorState, Simulator>(sampledState, _simulatorPtr, _searchTreePtr->_rootObservationNodePtr, _planningHorizon, 0);

    _looper.end();
  }
  _looper.log(results);

  selectedAction = this->selectAction<SimulatorState>(_searchTreePtr->_rootObservationNodePtr, false);
  
  // TODO: implement the tree node conversion
  return selectedAction;
}

template <class SimulatorState, class Simulator>
bool POMCPAtomicAgent<SimulatorState, Simulator>::_particleDepleted() const
{
  return _searchTreePtr->particleDepleted();
}

// the constructor
MixedPOMCPAtomicAgent::MixedPOMCPAtomicAgent(
  const std::string &agentID, 
  int numberOfActions, 
  int numberOfStepsToPlan, 
  const YAML::Node &agentParameters, 
  SelfImprovingSimulator* simulatorPtr
): AbstractPOMCPAtomicAgent(
  agentID,
  numberOfActions,
  numberOfStepsToPlan,
  agentParameters), 
  _simulatorPtr(simulatorPtr),
  metaAgent(this, agentParameters["Meta"])
{
  if (agentParameters["Simulator"]["ReplayBuffer"].IsDefined() && agentParameters["Simulator"]["ReplayBuffer"]["save"].IsDefined()) {
    _saveReplayBuffer = agentParameters["Simulator"]["ReplayBuffer"]["save"].as<bool>();
  } else {
    _saveReplayBuffer = false;
  }

  // for rollout
  _storeRolloutData = agentParameters["Meta"]["storeRolloutData"].as<bool>();
  _computeRolloutLoss = agentParameters["Meta"]["computeRolloutLoss"].as<bool>();

  // for action selection during tree search
  if (agentParameters["Meta"]["actionSelection"].as<std::string>() == "combined") {
    _actionSelectionMethod = COMBINED;
  } else if (agentParameters["Meta"]["actionSelection"].as<std::string>() == "independent") {
    _actionSelectionMethod = INDEPENDENT;
  } else {
    LOG(FATAL) << "The action selection method is not supported.";
  }

  _entropyEstimation = agentParameters["Meta"]["entropyEstimation"].as<bool>();
  if (_entropyEstimation == true) {
    if (agentParameters["Meta"]["entropyEstimationType"].as<std::string>() == "MLE") {
      _entropyEstimationType = MLE;
      LOG(FATAL) << "entropy estimation type MLE not supported";
    } else if (agentParameters["Meta"]["entropyEstimationType"].as<std::string>() == "NN") {
      _entropyEstimationType = NN;
    } else {
      _entropyEstimationType = EXACT;
    }
  } else {
    LOG(FATAL) << "entropy estimation type not supported";
  }

  // build up the two search trees
  _globalSearchTreePtr = new SearchTree<SelfImprovingSimulator::IndexedGlobalSimulatorState>(_numberOfActions);
  _IALSSearchTreePtr = new SearchTree<InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState>(_numberOfActions);
}

// the destructor
MixedPOMCPAtomicAgent::~MixedPOMCPAtomicAgent() {
  delete _globalSearchTreePtr;
  delete _IALSSearchTreePtr;
  delete _simulatorPtr;
}

bool MixedPOMCPAtomicAgent::_particleDepleted() const
{
  return _globalSearchTreePtr->particleDepleted() == true && _IALSSearchTreePtr->particleDepleted() == true;
}

// the observe function
void MixedPOMCPAtomicAgent::observe(int observation) {
  VLOG(3) << "[Agent " + _agentID + "]: observed " + std::to_string(observation) + ".";
  // if there is no more planning that needs to be done, then do nothing
  if (_planningHorizon == 0) {
    return;
  }

  bool particleDepletedAlready = this->_particleDepleted();

  // for global search tree
  bool globalSearchTreeParticleDepletedAlready = _globalSearchTreePtr->particleDepleted();
  VLOG(3) << "[Agent " + _agentID + "] [Global Search Tree]: number of particles in the tree left after the previous step: " << std::to_string(_globalSearchTreePtr->_rootObservationNodePtr->particles.size());
  this->_updateSearchTree(_globalSearchTreePtr, _simulatorPtr, _previousActionTaken, observation);
  if (globalSearchTreeParticleDepletedAlready == false && _globalSearchTreePtr->particleDepleted() == true) {
    LOG(INFO) << "Global search tree particle depleted with " << std::to_string(_planningHorizon) << " to go!";
  }

  // for IALS search tree
  bool IALSSearchTreeParticleDepletedAlready = _IALSSearchTreePtr->particleDepleted();
  VLOG(3) << "[Agent " + _agentID + "] [IALS Search Tree]: number of particles in the tree left after the previous step: " << std::to_string(_IALSSearchTreePtr->_rootObservationNodePtr->particles.size());
  this->_updateSearchTree(_IALSSearchTreePtr, _simulatorPtr, _previousActionTaken, observation);
  if (IALSSearchTreeParticleDepletedAlready == false && _IALSSearchTreePtr->particleDepleted() == true) {
    LOG(INFO) << "[Particle Depletion] IALS search tree particle depleted with " << std::to_string(_planningHorizon) << " to go!";
  }

  if (particleDepletedAlready == false && this->_particleDepleted() == true) {
    LOG(INFO) << "[Particle Depletion] Particle depletion has occurred with " << std::to_string(_planningHorizon) << " to go.";
  }

  VLOG(3) << "--------------------------------------------------";
}

void MixedPOMCPAtomicAgent::reset() {
  AbstractPOMCPAtomicAgent::reset();
  // destroy the previous search tree and build a new one
  _globalSearchTreePtr->reset();
  _IALSSearchTreePtr->reset();
  // reset the simulator - note that this will train the influence predictor
  _simulatorPtr->reset();
  // sample initial particles for the global search tree
  for (int i=0; i<=_numberOfParticles-1; i++) {
    _globalSearchTreePtr->_rootObservationNodePtr->particles.push_back(_simulatorPtr->sampleInitialGlobalState());
  }
  // sample initial particles for the IALS search tree
  for (int i=0; i<=_numberOfParticles-1; i++) {
    _IALSSearchTreePtr->_rootObservationNodePtr->particles.push_back(_simulatorPtr->sampleInitialIALSState());
  }
  metaAgent.resetEpisode();
}

void MixedPOMCPAtomicAgent::log(YAML::Node agentResults) const {
  metaAgent.log(agentResults);
  _simulatorPtr->train(agentResults);
}

int MixedPOMCPAtomicAgent::plan(YAML::Node &results) {

  int numGlobalParticles = 0;
  if (_globalSearchTreePtr->particleDepleted() == false) {
    numGlobalParticles = _globalSearchTreePtr->_rootObservationNodePtr->particles.size();
  }
  results["number_of_particles_before_simulation_global"].push_back(numGlobalParticles);
  int numIALSParticles = 0;
  if (_IALSSearchTreePtr->particleDepleted() == false) {
    numIALSParticles = _IALSSearchTreePtr->_rootObservationNodePtr->particles.size();
  }
  results["number_of_particles_before_simulation_IALS"].push_back(numIALSParticles);

  // preparation
  metaAgent.resetStep();
  
  int selectedAction;
  _looper.reset();
  while (_looper.finished() == false) {

    // warm up the influence predictor here?
    // we can ask the simulator to do that
    _simulatorPtr->warmUp();

    _looper.start();

    // pick the simulator 
    int simulatorType = metaAgent.pickSimulator();

    if (simulatorType == GS) {
      // sample a GS state
      SelfImprovingSimulator::IndexedGlobalSimulatorState sampledState = this->_globalSearchTreePtr->_rootObservationNodePtr->sampleParticle();
      // reserve the memory for the losses
      std::vector<float> crossEntropies;
      crossEntropies.reserve(_planningHorizon);
      std::vector<float> entropies;
      entropies.reserve(_planningHorizon);
      ObservationNode<InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState> *theOtherNodePtr = nullptr;
      if (_actionSelectionMethod == COMBINED) {
        theOtherNodePtr = _IALSSearchTreePtr->_rootObservationNodePtr;
      }
      // perform the simulation
      this->simulate(
        sampledState, 
        _simulatorPtr,
        _globalSearchTreePtr->_rootObservationNodePtr, 
        _planningHorizon, 
        0, 
        theOtherNodePtr, 
        &crossEntropies, 
        &entropies);
      // update meta statistics
      metaAgent.updateStep(crossEntropies, entropies);
    } else if (simulatorType == IALS) {
      // sample a IALS state
      InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState sampledState = this->_IALSSearchTreePtr->_rootObservationNodePtr->sampleParticle();
      // perform the simulation
      ObservationNode<SelfImprovingSimulator::IndexedGlobalSimulatorState> *theOtherNodePtr = nullptr;
      if (_actionSelectionMethod == COMBINED) {
        theOtherNodePtr = _globalSearchTreePtr->_rootObservationNodePtr;
      }
      this->simulate(
        sampledState, 
        _simulatorPtr,
        _IALSSearchTreePtr->_rootObservationNodePtr, 
        _planningHorizon, 
        0,
        theOtherNodePtr);
    }
    _looper.end();
  }
  _looper.log(results);
  metaAgent.wrapUpStep();

  selectedAction = this->selectAction<SelfImprovingSimulator::IndexedGlobalSimulatorState, InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState>(_globalSearchTreePtr->_rootObservationNodePtr, _IALSSearchTreePtr->_rootObservationNodePtr, false);
  return selectedAction;
}

void MixedPOMCPAtomicAgent::_initializeResultsYAML(YAML::Node results) {
  results["number_of_seconds_per_step"] = YAML::Node();
  results["number_of_simulations_per_step"] = YAML::Node();
  results["globalCounts"] = YAML::Node();
  results["globalValues"] = YAML::Node();
  results["IALSCounts"] = YAML::Node();
  results["IALSValues"] = YAML::Node();
  results["averageKLBonuses"] = YAML::Node();
  results["averageCrossEntropies"] = YAML::Node();
  results["averageEntropies"] = YAML::Node();
  results["number_of_particles_before_simulation_global"] = YAML::Node();
  results["number_of_particles_before_simulation_IALS"] = YAML::Node();
}

void MixedPOMCPAtomicAgent::save(const std::string &pathToResultsFolder) const {
  if (_saveReplayBuffer == true) {
      _simulatorPtr->getReplayBufferPtr()->save(pathToResultsFolder);
  } 
}