#ifndef AGENTCOMPONENT_HPP_
#define AGENTCOMPONENT_HPP_

#include "glog/logging.h"
#include "agents/AtomicAgent.hpp"
#include <memory>
#include "yaml-cpp/yaml.h"
#include <ctime>

// an abstract class
class AgentComponent {
  public:
    AgentComponent() {

    }
    virtual ~AgentComponent() {
      
    }
    virtual void reset() = 0;
    virtual void act(std::map<std::string, int> &action, YAML::Node &agentsYAMLNode) = 0;
    virtual void observe(std::map<std::string, int> &observation) = 0;
    virtual void log(YAML::Node results) const = 0;
    virtual void save(const std::string &pathToResultsFolder) const = 0;
};

// an agent component which consists of only atomic agents
class SimpleAgentComponent: public AgentComponent {
  public:
    SimpleAgentComponent(std::map<std::string, AtomicAgent*> &atomicAgents): AgentComponent() {
      for (auto &[agentID, agentPtr]: atomicAgents) {
        _atomicAgents[agentID] = std::unique_ptr<AtomicAgent>(agentPtr);
      }
      LOG(INFO) << "A simple Agent component consisting of " << atomicAgents.size() << " agents has been built.";
    }
    ~SimpleAgentComponent() {
      // atomic agents will be deleted automatically because they are wrapped by unique pointers
    }

    // the agent component resets the interal states of the atomic agents
    void reset() {
      for (auto &[key, val]: _atomicAgents) {
        val->reset();
      }
      VLOG(1) << "Agent component has been reset.";
    }
    
    void act(std::map<std::string, int> &action, YAML::Node &results) {
      for (const auto &[actionID, val]: _atomicAgents) {
        action[actionID] = val->act(results[actionID]);
      }
    }
    
    // the agent component receives the joint observation from the environment
    void observe(std::map<std::string, int> &observation) {
      for (const auto &[key, val]: _atomicAgents) {
        val->observe(observation[key]);
      }
    }

    void log(YAML::Node results) const override {
      for (const auto &[agentID, agent]: _atomicAgents) {
        agent->log(results[agentID]);
      }
    }

    void save(const std::string &pathToResultsFolder) const {
      for (const auto &[agentID, agent]: _atomicAgents) {
        agent->save(pathToResultsFolder);
      }
    }

private:
  std::map<std::string, std::unique_ptr<AtomicAgent>> _atomicAgents;
};

#endif