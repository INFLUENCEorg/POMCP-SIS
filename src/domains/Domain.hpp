#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include "agents/AtomicAgent.hpp"
#include "agents/PlanningAgent.hpp"
#include "agents/AgentComponent.hpp"
#include "Environment.hpp"
#include "dbns/TwoStageDynamicBayesianNetwork.hpp"
#include "Utils.hpp"

// Assumptions & Data types:
// * an action is an integer. mutli agent action is a map of agentID(string)->action.
// * an observation is an integer. multi agent observation is a map of agentID(string)->observation.
// * a reward is a float. multi agent reward is a map of agentID(string)->reward.
// * a done is a boolean. multi agent done is a boolean.
// * different simulators use different types of states.

// Domain is a central component of an experiment. 
// It provides the functionality to build up the environment and the agents. 
// It builds up the dynamic bayesian network and provides high-level interface for the environment and agents to utilize the DBN,

class Domain {
  public:

    Domain(const YAML::Node &parameters);

    virtual ~Domain();

    // factory method: make the environment
    Environment makeEnvironment() const;

    // factory method: make agent component
    AgentComponent* makeAgentComponent() const;

    // factory method: make agent simulator, will be used by the global simulator 
    virtual AtomicAgentSimulator *makeAtomicAgentSimulator(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters)const; 

    // get method: get the discount factor
    float getDiscountFactor() const { return _discountFactor; }

    // get method: get the number of agents
    int getNumberOfAgents() const{  return _listOfAgentIDs.size(); }

    // get method: get the list of agent ids
    const std::vector<std::string> &getListOfAgentIDs() const {return _listOfAgentIDs;} 
    
    // get method: get the list of agent actions
    const std::map<std::string, int> &getListOfAgentActions() const {return _numberOfActions;}
    
    // get method: get number of steps to plan
    int getNumberOfStepsToPlan() const { return _numberOfStepsToPlan; }

    // get method: get domain name
    std::string getDomainName() const { return _domainName; }

    // get method: get the DBN
    const TwoStageDynamicBayesianNetwork &getDBN() const { return _DBN; }

    // simulation method: sample initial state
    virtual std::map<std::string, int> sampleInitialState() const;
    
    // simulation method: simulate one-step transition
    virtual void step(std::map<std::string, int> &state, std::map<std::string, int> &action, std::map<std::string, int> &observation, std::map<std::string, float> &reward, bool &done, const std::string &samplingMode) const;

    void constructLocalModel(const std::string &agentID, std::vector<std::string> &localStates, std::vector<std::string> &sourceFactors, std::vector<std::string> &destinationFactors, std::vector<std::string> &DSeparationVariablesPerStep) const;

    const YAML::Node &getParameters() const {
      return _parameters;
    }

protected:
  TwoStageDynamicBayesianNetwork _DBN;
  const YAML::Node &_parameters;
  std::string _domainName;
  int _numberOfAgents;
  int _numberOfEnvironmentStates;
  int _numberOfStepsToPlan;
  float _discountFactor;
  std::vector<std::string> _listOfAgentIDs;
  std::map<std::string, int> _numberOfActions;

  // factory method: make agent
  virtual AtomicAgent *makeAtomicAgent(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const;

  // (DOMAIN-SPECIFIC) factory method: make domain-specific agent
  virtual AtomicAgent *makeDomainSpecificAtomicAgent(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const = 0;

  // (DOMAIN-SPECIFIC) factory method: make domain-specific agent simulator
  virtual AtomicAgentSimulator *makeDomainSpecificAtomicAgentSimulator(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const = 0;
};

#endif
