#ifndef Grid_Traffic_DOMAIN_HPP_
#define Grid_Traffic_DOMAIN_HPP_

#include "domains/Domain.hpp"
#include "GridTrafficAtomicAgent.hpp"

class GridTrafficDomain: public Domain {

  private:
    int _obsLength;

  public:
    
    GridTrafficDomain(const YAML::Node &parameters) : Domain(parameters) {
      _obsLength = parameters["GridTraffic"]["obsLength"].as<int>();
    }
    
    AtomicAgentSimulator *makeDomainSpecificAtomicAgentSimulator(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgentSimulator *agentSimulatorPtr = nullptr;
      if (agentType == "Simple2") {
        agentSimulatorPtr = new GridTrafficSimple2AtomicAgentSimulator(_obsLength);
      } else if (agentType == "Pattern") {
        agentSimulatorPtr = new GridTrafficPatternAtomicAgentSimulator(agentParameters["freq"].as<int>());
      } 
      return agentSimulatorPtr;
    }

    AtomicAgent *makeDomainSpecificAtomicAgent(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgent *atomicAgentPtr = nullptr;
      if (agentType == "Simple2") {
        atomicAgentPtr = new GridTrafficSimple2AtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters, _obsLength);
      } else if (agentType == "Pattern") {
        atomicAgentPtr = new GridTrafficPatternAtomicAgent(
          agentID, 
          _numberOfActions.at(agentID), 
          _numberOfStepsToPlan, agentParameters,
          agentParameters["freq"].as<int>());
      } 
      return atomicAgentPtr;
    }
};

#endif