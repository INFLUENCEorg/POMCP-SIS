#ifndef GRAB_A_CHAIR_DOMAIN_HPP_
#define GRAB_A_CHAIR_DOMAIN_HPP_

#include "domains/Domain.hpp"
#include "GrabAChairAtomicAgent.hpp"

class GrabAChairDomain: public Domain {

  public:
    
    GrabAChairDomain(const YAML::Node &parameters) : Domain(parameters) {}
    
    AtomicAgentSimulator *makeDomainSpecificAtomicAgentSimulator(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgentSimulator *agentSimulatorPtr = nullptr;
      if (agentType[0] == 'A') {
        agentSimulatorPtr = new DeterministicAtomicAgentSimulator(_numberOfActions.at(agentID), (int)(agentType[1] - (int)'0'));
      } else if (agentType.rfind("Pattern", 0) == 0) {
        agentSimulatorPtr = new GrabAChairPatternAtomicAgentSimulator((int)(agentType[7]-(int)'0'));
      } else if (agentType == "Count") {
        agentSimulatorPtr = new GrabAChairCountBasedAtomicAgentSimulator();
      } else if (agentType == "Happy") {
        agentSimulatorPtr = new GrabAChairHappyAtomicAgentSimulator();
      } else if (agentType == "Sad") {
        agentSimulatorPtr = new GrabAChairSadAtomicAgentSimulator();
      } 
      return agentSimulatorPtr;
    }

    AtomicAgent *makeDomainSpecificAtomicAgent(const std::string &agentID, const std::string &agentType, YAML::Node agentParameters) const override {
      AtomicAgent *atomicAgentPtr = nullptr;
      if (agentType == "Random") {
        atomicAgentPtr = new RandomAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters);
      } else if (agentType == "A0") {
        atomicAgentPtr = new DeterministicAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters, 0);
      } else if (agentType == "A1") {
        atomicAgentPtr = new DeterministicAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters, 1);
      } else if (agentType.rfind("Pattern", 0) == 0) {
        atomicAgentPtr = new GrabAChairPatternAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters, (int)(agentType[7]-'0'));
      } else if (agentType == "Count") {
        atomicAgentPtr = new GrabAChairCountBasedAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters);
      } else if (agentType == "Happy") {
        atomicAgentPtr = new GrabAChairHappyAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters);
      } else if (agentType == "Sad") {
        atomicAgentPtr = new GrabAChairSadAtomicAgent(agentID, _numberOfActions.at(agentID), _numberOfStepsToPlan, agentParameters);
      }
      return atomicAgentPtr;
    }
};

#endif