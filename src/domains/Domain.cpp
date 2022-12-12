#include "Domain.hpp"
#include "yaml-cpp/yaml.h"
#include <experimental/random>
#include <math.h>
#include "agents/PlanningAgentSimulator.hpp"

Domain::Domain(const YAML::Node &parameters): 
_parameters(parameters), 
_domainName(_parameters["General"]["domain"].as<std::string>()) {
  
  _DBN = TwoStageDynamicBayesianNetwork(parameters[_domainName]["2SDBNYamlFilePath"].as<std::string>());

  // loading agent IDs
  for(YAML::const_iterator it=_parameters["AgentComponent"].begin(); it!=_parameters["AgentComponent"].end(); ++it) {
    std::string agentID = it->first.as<std::string>();
    _listOfAgentIDs.push_back(agentID);
  }
  VLOG(1) << "list of agents: " + PrintUtils::vectorToString(_listOfAgentIDs);

  // initializing the DBN
  _DBN.computeFullSamplingOrder(); 

  // get domain parameters
  _numberOfActions = _DBN.getListOfAgentActions();
  _numberOfAgents = _listOfAgentIDs.size();
  _numberOfStepsToPlan = _parameters["General"]["horizon"].as<int>();
  _numberOfEnvironmentStates = _DBN.getNumberOfStates();
  _discountFactor = _parameters["General"]["discountFactor"].as<float>();

  LOG(INFO) << _domainName << " has been initialized.";
}

AtomicAgentSimulator *Domain::makeAtomicAgentSimulator(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const {
  
  AtomicAgentSimulator *agentSimulatorPtr = nullptr;
  
  if (agentType == "Random") {
    agentSimulatorPtr = new RandomAtomicAgentSimulator(_numberOfActions.at(agentID));
  } else {
    agentSimulatorPtr = this->makeDomainSpecificAtomicAgentSimulator(agentID, agentType, agentParameters);
  }
  
  if (agentSimulatorPtr == nullptr) {
    LOG(FATAL) << "Agent type " << agentType << " not supported.";
  }
  return agentSimulatorPtr;
}

AtomicAgent *Domain::makeAtomicAgent(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const {  
  AtomicAgent *atomicAgentPtr = nullptr;
  agentParameters["Rollout"]["discountFactor"] = _discountFactor;
  if (agentType == "Random") {
    atomicAgentPtr = new RandomAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters);
  } else if (agentType == "POMCP") {
    std::string simulatorType = agentParameters["Simulator"]["Type"].as<std::string>();
    if (simulatorType == "Global") {
      GlobalSimulator *simulatorPtr = new GlobalSimulator(
        agentID, 
        this, 
        _parameters["AgentComponent"]
      );
      atomicAgentPtr = new POMCPAtomicAgent<GlobalSimulator::GlobalSimulatorState, GlobalSimulator>(
        agentID, 
        _numberOfActions.at(agentID), 
        _numberOfStepsToPlan, 
        agentParameters,
        simulatorPtr
      );
    } else if (simulatorType == "Local") {
      InfluenceAugmentedLocalSimulator *simulatorPtr = new InfluenceAugmentedLocalSimulator(
        agentID, 
        this, 
        agentParameters["Simulator"]
      );
      simulatorPtr->initializeInfluencePredictor(agentParameters["Simulator"]);
      atomicAgentPtr = new POMCPAtomicAgent<InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState, InfluenceAugmentedLocalSimulator>(
        agentID, 
        _numberOfActions.at(agentID), 
        _numberOfStepsToPlan, 
        agentParameters,
        simulatorPtr
      );
    } else {
      std::string message = "Unsupported simulator type: " + simulatorType + ".";
      LOG(FATAL) << message;
      atomicAgentPtr = nullptr;
    }   
  } else if (agentType == "MixedPOMCP" && agentParameters["Simulator"]["Type"].as<std::string>() == "SelfImproving") {
      SelfImprovingSimulator *simulatorPtr = new SelfImprovingSimulator(
        agentID,
        this,
        agentParameters["Simulator"],
        _parameters["AgentComponent"]
      );
      atomicAgentPtr = new MixedPOMCPAtomicAgent(
        agentID,
        _numberOfActions.at(agentID), 
        _numberOfStepsToPlan, 
        agentParameters,
        simulatorPtr
      );
    } else {
    atomicAgentPtr = this->makeDomainSpecificAtomicAgent(agentID, agentType, agentParameters);
  }

  if (atomicAgentPtr == nullptr) {
    std::string message = "Agent Type " + agentType + " is not supported.";
    LOG(FATAL) << message;
  }

  return atomicAgentPtr;
}

Environment Domain::makeEnvironment() const{
  return Environment(*this);
}

AgentComponent *Domain::makeAgentComponent() const {
  VLOG(1) << "Making the agent component.";
  std::map<std::string, AtomicAgent*> atomicAgents;
  for (auto &[agentID, numberOfActions]: this->getListOfAgentActions()) {
    std::string agentType = _parameters["AgentComponent"][agentID]["Type"].as<std::string>();
    atomicAgents[agentID] = this->makeAtomicAgent(agentID, agentType, _parameters["AgentComponent"][agentID]);
    VLOG(1) << "Agent " << agentID << " of type " << agentType << " has been built.";
  }
  return new SimpleAgentComponent(atomicAgents);
}

Domain::~Domain(){
  VLOG(1) << "Domain deleted.";
}

std::map<std::string, int> Domain::sampleInitialState() const {
  return _DBN.sampleInitialState();
}

void Domain::step(std::map<std::string, int> &state, std::map<std::string, int> &action, std::map<std::string, int> &observation, std::map<std::string, float> &reward, bool &done, const std::string &samplingMode) const {

  // read actions
  for (const auto &agentID: _listOfAgentIDs) {
    state["a"+agentID] = action[agentID];
  }
  
  // perform one step sampling in the two stage dynamic bayesian network
  _DBN.step(state, samplingMode); 

  VLOG(5) << "Finished one step in the bayesian network.";

  // update state, observation, reward and done
  for (const auto &agentID: _listOfAgentIDs) {
    reward[agentID] = _DBN.getValueOfVariableFromIndex("r" + agentID, state);
    observation[agentID] = _DBN.getValueOfVariableFromIndex("o" + agentID, state);
  }

  done = false;
}

void Domain::constructLocalModel(const std::string &agentID, std::vector<std::string> &localStates, std::vector<std::string> &sourceFactors, std::vector<std::string> &destinationFactors, std::vector<std::string> &DSeparationVariablesPerStep) const {
  this->_DBN.constructLocalModel(agentID, localStates, sourceFactors, destinationFactors, DSeparationVariablesPerStep);
  if (_domainName == "GridTraffic") {
    DSeparationVariablesPerStep.clear();
    for (const std::string &varName: destinationFactors) {
      DSeparationVariablesPerStep.push_back(StringUtils::removeLastPrime(varName));
    }
    LOG(INFO) << "D separation variables reset to " << PrintUtils::vectorToTupleString(DSeparationVariablesPerStep);
  }  
}