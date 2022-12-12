#ifndef Influence_Cycle_DOMAIN_HPP_
#define Influence_Cycle_DOMAIN_HPP_

#include "domains/Domain.hpp"

class InfluenceCycleDomain: public Domain {

  public:
    
    InfluenceCycleDomain(const YAML::Node &parameters) : Domain(parameters) {
    }
    
    AtomicAgentSimulator *makeDomainSpecificAtomicAgentSimulator(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgentSimulator *agentSimulatorPtr = nullptr;
      return agentSimulatorPtr;
    }

    AtomicAgent *makeDomainSpecificAtomicAgent(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgent *atomicAgentPtr = nullptr;
      return atomicAgentPtr;
    }
};

#endif