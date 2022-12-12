#ifndef ATOMIC_AGENT_HPP_
#define ATOMIC_AGENT_HPP_

#include "glog/logging.h"
#include "yaml-cpp/yaml.h"
#include <experimental/random>
#include <queue>
#include "Utils.hpp"
#include <math.h>
#include <memory>

class AtomicAgent {
  public:
    AtomicAgent(const std::string &agentID, int numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters) {
      _numberOfActions = numberOfActions;
      _numberOfStepsToPlan = numberOfStepsToPlan;
      _agentID = agentID;
      _parameters = parameters;
    }
    // clear and reset the AOH
    virtual void reset() {
      _AOH.clear();
      _AOH.resize(1 + 2 * _numberOfStepsToPlan);
      _AOH[0] = 1; // restart the AOH
    }
    virtual int act(YAML::Node agentsYAMLNode) = 0;
    virtual void observe(int observation) = 0;
    virtual void log(YAML::Node agentResults) const {};
    virtual void save(const std::string &pathToResultsFolder) const {};
  protected:
    int _numberOfActions;
    int _numberOfStepsToPlan;
    YAML::Node _parameters;
    std::string _agentID;
    std::vector<int> _AOH;
};

class AtomicAgentSimulator {
  public:
    AtomicAgentSimulator() {}
    // the key function that reads in a history and gives an action and updates the history with the action taken
    virtual int step(const std::vector<int>::iterator &it) = 0;
    virtual int step(const std::vector<int>::iterator &it, float &entropy) = 0;
    // this function should be same for every agent simulator and atomic agents that are not planning agents
    virtual void observe(const std::vector<int>::iterator &it, int &observation) {
      *(it+(*it)) = observation;
      *it += 1;
    }
};

class DeterministicAtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    DeterministicAtomicAgentSimulator(const int &numberOfActions, int action): _action(action) {

    }
    virtual int step(const std::vector<int>::iterator &it) {
      *(it+(*it)) = _action;
      *it += 1;
      return _action;
    }
    int step(const std::vector<int>::iterator &it, float &entropy) {
      entropy = 0.0;
      return this->step(it);
    }
  private:
    int _action;
};

class DeterministicAtomicAgent: public AtomicAgent, private DeterministicAtomicAgentSimulator {
  public:
    DeterministicAtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters, int action): DeterministicAtomicAgentSimulator(numberOfActions, action), AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {}
    int act(YAML::Node agentYAMLNode) {
      return DeterministicAtomicAgentSimulator::step(_AOH.begin());
    }
    virtual void observe(int observation) {
      DeterministicAtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};

class RandomAtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    RandomAtomicAgentSimulator(const int &numberOfActions): AtomicAgentSimulator(), _numberOfActions(numberOfActions) {}
    int step() { 
      return std::experimental::randint(0, _numberOfActions-1);
    }
    int step(const std::vector<int>::iterator &it) {
      int action = step();
      *(it+(*it)) = action;
      *it += 1;
      return action;
    }
    int step(const std::vector<int>::iterator &it, float &entropy) override {
      entropy = std::log(_numberOfActions);
      return this->step(it);
    }
  private:
    int _numberOfActions;
};

class RandomAtomicAgent: public AtomicAgent, private RandomAtomicAgentSimulator {
  public:
    RandomAtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters): RandomAtomicAgentSimulator(numberOfActions), AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {}
    int act(YAML::Node agentsYAMLNode) {
      return RandomAtomicAgentSimulator::step(_AOH.begin());
    }
    virtual void observe(int observation) {
      RandomAtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};

#endif