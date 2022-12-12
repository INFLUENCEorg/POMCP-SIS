#ifndef FIRE_FIGHTER_DOMAIN_HPP_
#define FIRE_FIGHTER_DOMAIN_HPP_

#include "domains/Domain.hpp"
#include "FireFighterAtomicAgent.hpp"

class FireFighterDomain: public Domain {

  public:
    
    FireFighterDomain(const YAML::Node &parameters) : Domain(parameters) {}
    
    AtomicAgentSimulator* makeDomainSpecificAtomicAgentSimulator(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgentSimulator *agentSimulatorPtr;
      if (agentType == "Random") {
        agentSimulatorPtr = new RandomAtomicAgentSimulator(_numberOfActions.at(agentID));
      } else if (agentType == "Naive") {
        agentSimulatorPtr = new FireFighterNaiveAtomicAgentSimulator();
      } else {
        LOG(FATAL) << "Agent type " << agentType << " not supported.";
      }
      return agentSimulatorPtr;
    }

    AtomicAgent* makeDomainSpecificAtomicAgent(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgent *atomicAgentPtr = nullptr;
      if (agentType == "Naive") {
        atomicAgentPtr = new FireFighterNaiveAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters);
      }
      return atomicAgentPtr;
    }
};

#endif