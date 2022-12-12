#ifndef TWO_STAGE_DYNAMIC_BAYESIAN_NETWORK_HPP_
#define TWO_STAGE_DYNAMIC_BAYESIAN_NETWORK_HPP_

#include "TwoStageDynamicBayesianVariable.hpp"
#include <iostream>
#include <random>
#include "yaml-cpp/yaml.h"
#include "glog/logging.h"
#include <memory>
#include "Utils.hpp"
#include <algorithm>
#include <json.hpp>
#include <fstream>
#include <ctime>
#include <map>

// the two stage dynamic bayesian network
class TwoStageDynamicBayesianNetwork {
  public:

    TwoStageDynamicBayesianNetwork() {}

    // the constructor
    TwoStageDynamicBayesianNetwork(const std::string &yamlFilePath){
      LOG(INFO) << "Loading " << yamlFilePath << ".";
      clock_t begin = std::clock();
      std::ifstream ifs(yamlFilePath);
      std::string content;
      content.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
      YAML::Node config = YAML::Load(content);
      double elapsed_seconds = double(std::clock()-begin) / CLOCKS_PER_SEC;
      LOG(INFO) << "Yaml file loaded after " << std::to_string(elapsed_seconds) << " seconds. Constructing the DBN ...";
      for (YAML::const_iterator it = config.begin(); it != config.end(); ++it){
        std::string key = it->first.as<std::string>();
        _twoStageDynamicBayesianNetworkVariables.insert(std::pair<std::string, TwoStageDynamicBayesianNetworkVariable>(key, TwoStageDynamicBayesianNetworkVariable(key, config[key], config)));
        if (_twoStageDynamicBayesianNetworkVariables.at(key).isStateVariable() == true) {
          _stateVariables.push_back(key);
        } 
      }
      LOG(INFO) << "Two stage dynamic bayesian network has been built.";
    }

    // the destructor
    ~TwoStageDynamicBayesianNetwork() {
    }

    // for influence-augmented local simulator
    void constructLocalModel(const std::string &agentID, std::vector<std::string> &localStates, std::vector<std::string> &influenceSourceVariables, std::vector<std::string> &influenceDestinationVariables, std::vector<std::string> &DSeparationVariablesPerStep) const {

      std::set<std::string> _localFactors;
      std::set<std::string> _influenceSourceVariables;
      std::set<std::string> _influenceDestinationVariables;
      std::set<std::string> _DSeparationVariablesPerStep; // the final d separation set is a history of this set
      
      // names of observation and reward variables
      std::string observationFactorName = "o" + agentID;
      std::string rewardFactorName = "r" + agentID;
      LOG(INFO) << "Identifying local state varibles.";
      LOG(INFO) << observationFactorName;
      LOG(INFO) << rewardFactorName;

      // find local factors = local states + action
      for (const std::string &parentFactorName: _twoStageDynamicBayesianNetworkVariables.at(observationFactorName).getListOfParents()) {
        LOG(INFO) << parentFactorName;
        _localFactors.insert(StringUtils::removeLastPrime(parentFactorName));
      }
      _localFactors.insert("a"+agentID);
      for (const std::string &parentFactorName: _twoStageDynamicBayesianNetworkVariables.at(rewardFactorName).getListOfParents()) {
        LOG(INFO) << parentFactorName;
        _localFactors.insert(StringUtils::removeLastPrime(parentFactorName));
      }
      LOG(INFO) << PrintUtils::setToTupleString(_localFactors);
      std::vector<std::string> localFactors = ContainerUtils::setToVector(_localFactors);
      std::sort(localFactors.begin(), localFactors.end(), _factorComparator);
      LOG(INFO) << "Local state variables and actions: " << PrintUtils::vectorToTupleString(localFactors);

      // find influence source variables and influence destination variables
      LOG(INFO) << "Identifying influence source state variables and influence destination state variables.";
      for (const auto &localFactorName: _localFactors) {
        bool hasLinkFromOutside = false;
        for (const std::string &localFactorParentName: _twoStageDynamicBayesianNetworkVariables.at(StringUtils::addLastPrime(localFactorName)).getListOfParents()) {
          if (_localFactors.find(localFactorParentName) == _localFactors.end() && _localFactors.find(StringUtils::addLastPrime(localFactorParentName)) == _localFactors.end()) {
            _influenceSourceVariables.insert(StringUtils::removeLastPrime(localFactorParentName));
            hasLinkFromOutside = true;
          }
        }
        if (hasLinkFromOutside == true) {
          _influenceDestinationVariables.insert(StringUtils::addLastPrime(localFactorName));
        }
      }
      _DSeparationVariablesPerStep.insert(_localFactors.begin(), _localFactors.end()); // for now
      
      // sort influence source variables, influence destination variables and d separation variables per step
      influenceSourceVariables = ContainerUtils::setToVector(_influenceSourceVariables);
      std::sort(influenceSourceVariables.begin(), influenceSourceVariables.end(), _factorComparator);
      LOG(INFO) << "Influence source state variables: " << PrintUtils::vectorToTupleString(influenceSourceVariables);
      influenceDestinationVariables = ContainerUtils::setToVector(_influenceDestinationVariables);
      std::sort(influenceDestinationVariables.begin(), influenceDestinationVariables.end(), _factorComparator);
      LOG(INFO) << "Influence destination state variables: " << PrintUtils::vectorToTupleString(influenceDestinationVariables);
      DSeparationVariablesPerStep = ContainerUtils::setToVector(_DSeparationVariablesPerStep);
      std::sort(DSeparationVariablesPerStep.begin(), DSeparationVariablesPerStep.end(), _factorComparator);
      LOG(INFO) << "D Separation Set Per Step: " << PrintUtils::vectorToTupleString(DSeparationVariablesPerStep);
      for (auto &factorName: localFactors) {
        if (factorName[0] != 'a' && factorName[0] != 'o') {
          localStates.push_back(factorName);
        }
      }
      std::sort(localStates.begin(), localStates.end(), _factorComparator);
      LOG(INFO) << "Local state variables: " << PrintUtils::vectorToTupleString(localStates);
      
      // compute sampling order for influence-augmented local model
      std::set<std::string> setIn = std::set<std::string>(_localFactors);
      for (auto &factor: influenceSourceVariables) setIn.insert(factor);
      std::set<std::string> setOut = _getStateFactorsNextStage(_localFactors);
      setOut.insert(observationFactorName);
      setOut.insert(rewardFactorName);
      LOG(INFO) << "Inputs to PGM: " << PrintUtils::setToTupleString(setIn);
      LOG(INFO) << "Outputs from PGM: " << PrintUtils::setToTupleString(setOut);
      computeSamplingOrder(setIn, setOut, "local");
      LOG(INFO) << "Local model has been constructed.";
    }

    // compute a sampling order given set of input variables and output variables and assign a name (sampling mode) to it
    void computeSamplingOrder(const std::set<std::string> &setOfInputVariables, const std::set<std::string> &setOfOutputVariables, const std::string &samplingMode) const {
      std::set<std::string> toSamplePool;
      std::set<std::string> sampledPool;
      toSamplePool.insert(setOfOutputVariables.begin(), setOfOutputVariables.end());
      sampledPool.insert(setOfInputVariables.begin(), setOfInputVariables.end());
      LOG(INFO) << "Computing sampling order with inputs: " + PrintUtils::setToTupleString<std::string>(sampledPool);
      LOG(INFO) << "and outputs: " + PrintUtils::setToTupleString<std::string>(toSamplePool);
      _samplingOrders[samplingMode] = std::vector<std::string>();
      while (toSamplePool.size() > 0){
        for (const auto& var: toSamplePool){
          bool allParentsAreSampled = true;
          for (const auto& parentName: _twoStageDynamicBayesianNetworkVariables.at(var).getListOfParents()){
            if (sampledPool.find(parentName) == sampledPool.end()){
              allParentsAreSampled = false;
              toSamplePool.insert(parentName);
            }
          }
          if (allParentsAreSampled == true){
              _samplingOrders[samplingMode].push_back(var);
              sampledPool.insert(var);
              break;
          }
        }
        toSamplePool.erase(_samplingOrders[samplingMode].back());
      }
      LOG(INFO) << "sampling order: " + PrintUtils::vectorToString<std::string>(_samplingOrders.at(samplingMode));
    }

    void computeFullSamplingOrder() const {
      
      std::set<std::string> setIn;
      std::set<std::string> setOut;

      for (auto const & [varName, variable]: _twoStageDynamicBayesianNetworkVariables) {
        if (varName[0] == 'a') {
          setIn.insert(varName);
        } else if (varName[0] == 'o' || varName[0] == 'r') {
          setOut.insert(varName);
        } else {
          if (StringUtils::lastBitIsPrime(varName) == true) {
            setOut.insert(varName);
          } else {
            setIn.insert(varName);
          }
        }
      }      
      computeSamplingOrder(setIn, setOut, "full");
    }

    // simulation method
    void step(std::map<std::string, int> &state, const std::string &samplingMode) const {
      for (auto const &varName: _samplingOrders[samplingMode]){
        int sampledValue = _twoStageDynamicBayesianNetworkVariables.at(varName).sample(state);
        state[varName] = sampledValue;
      }
      // update state variables
      for (auto const &key: _samplingOrders[samplingMode]) {
        if (key[0] == 'x') {
          state.at(key.substr(0, key.size()-1)) = state.at(key);
        }
      }
      state["t"] += 1;
    }

    // simulation method
    void stepWithEntropy(std::map<std::string, int> &state, const std::string &samplingMode, const std::vector<std::string> &influenceSourceStates, std::map<std::string, float> &exactInfluenceSourceEntropy) const {
      
      assert(samplingMode == "full");

      for (auto const &varName: _samplingOrders[samplingMode]){
        auto sampledValue = _twoStageDynamicBayesianNetworkVariables.at(varName).sample(state);
        state[varName] = sampledValue ;

        if (varName[0] != 'a' && std::find(influenceSourceStates.begin(), influenceSourceStates.end(), StringUtils::removeLastPrime(varName)) != influenceSourceStates.end()) {
          exactInfluenceSourceEntropy[varName] = _twoStageDynamicBayesianNetworkVariables.at(varName).getEntropy(state);
        }
      }
      // update state variables
      for (auto const &key: _samplingOrders[samplingMode]) {
        if (key[0] == 'x') {
          state.at(key.substr(0, key.size()-1)) = state.at(key);
        }
      }
      state["t"] += 1;

    }
      
    // simulation method: sample an initial state
    std::map<std::string, int> sampleInitialState() const {
      std::map<std::string, int> initialMap;
      for (const std::string &key: _stateVariables) {
        initialMap[key] = _twoStageDynamicBayesianNetworkVariables.at(key).sampleInitialValue();
      }
      initialMap["t"] = 0;
      return initialMap;
    }

    // get method
    float getValueOfVariableFromIndex(const std::string &variableName, const std::map<std::string, int> &state) const {
      return _twoStageDynamicBayesianNetworkVariables.at(variableName).getValueFromIndex(state.at(variableName));
    }

    // get method
    int getNumberOfStates() const {
      int count = 0;
      for (auto &[key, val]: _twoStageDynamicBayesianNetworkVariables) {
        if (key[0] == 'x' && StringUtils::lastBitIsPrime(key) == false) {
          count += 1;
        }
      }
      return count;
    }

    // get method: get variable by name
    const TwoStageDynamicBayesianNetworkVariable &getVariable(const std::string &varName) const {
      return _twoStageDynamicBayesianNetworkVariables.at(varName);
    }

    // get method: get list of agent actions
    std::map<std::string, int> getListOfAgentActions() const {
      std::map<std::string, int> numberOfActions;
      for (const auto &[key, var]: _twoStageDynamicBayesianNetworkVariables) {
        if (key[0] == 'a') {
          auto agentID = key.substr(1);
          int numberOfValues = var.getNumberOfValues();
          numberOfActions[agentID] = numberOfValues;
        }
      }
      return numberOfActions;
    }

    // get method: get the list of state variables
    const std::vector<std::string> &getStateVariables() const {
      return _stateVariables;
    }

  private:
    std::map<std::string, TwoStageDynamicBayesianNetworkVariable> _twoStageDynamicBayesianNetworkVariables;
    std::vector<std::string> _stateVariables;
    mutable std::map<std::string, std::vector<std::string>> _samplingOrders;
    static bool _factorComparator(const std::string &a_, const std::string &b_) {
      auto a = StringUtils::removeLastPrime(a_);
      auto b = StringUtils::removeLastPrime(b_);
      if (a[0] == 'x' && b[0] != 'x') {
        return 1;
      } else if (b[0] == 'x' && a[0] != 'x') {
        return 0;
      } else if (a[0] == 'a' && b[0] != 'a') {
        return 1;
      } else if (b[0] == 'a' && a[0] != 'a') {
        return 0;
      } else if (a[0] == 'o' && b[0] != 'o') {
        return 1;
      } else if (b[0] == 'o' && a[0] != 'o') {
        return 0;
      } else {
        try {
          // find the common prefix of two strings
          int idx = 0;
          while (true) {
            if (idx > a.size()-1 || idx > b.size()-1) {
              break;
            } else if (a[idx] != b[idx]) {
              break;
            } else {
              idx += 1;
            }
          }
          int aN = std::stoi(a.substr(idx));
          int bN = std::stoi(b.substr(idx));
          if (aN < bN) {
            // means larger second
            return 1;
          } else {
            // means smaller first
            return 0;
          }
        } catch (std::invalid_argument &e) {
          int result = a.compare(b);
          if (result < 0) {
            return 1;
          } else {
            return 0;
          }
        }
      }
    }
    std::set<std::string> _getStateFactorsNextStage(const std::set<std::string> &factors) const {
      std::set<std::string> set;
      for (auto &factor: factors) {
        if (factor[0] != 'a') {
          set.insert(StringUtils::addLastPrime(factor));
        }
      return set;
    }

};

};

#endif
