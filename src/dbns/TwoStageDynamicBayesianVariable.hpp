#ifndef TWO_STAGE_DYNAMIC_BAYESIAN_VARIABLE_HPP_
#define TWO_STAGE_DYNAMIC_BAYESIAN_VARIABLE_HPP_

#include <cmath>
#include <experimental/random>
#include <map>
#include "Utils.hpp"
#include <random>
#include <glog/logging.h>

#define CPT 0
#define SUM 1
#define EXPSUM 2
#define NOISYEXPSUM 3

class TwoStageDynamicBayesianNetworkVariable {
  public:

    std::string name;

    TwoStageDynamicBayesianNetworkVariable(const std::string &name, YAML::Node info, YAML::Node allInfo): 
    name(name) {
      
      _listOfParents = info["parents"].as<std::vector<std::string>>();
      if (info["values"].IsDefined()) {
        _listOfValues = info["values"].as<std::vector<float>>();
      }
      _numberOfValues = _listOfValues.size();

      if (_listOfParents.size() != 0) {
        _mode = CPT;
        if (info["mode"].IsDefined() == true) {
          std::string modeStr = info["mode"].as<std::string>();
          if (modeStr == "SUM") {
            _mode = SUM;
          } else if (modeStr == "EXPSUM") {
            _mode = EXPSUM;
          } else if (modeStr == "NOISYEXPSUM") {
            _mode = NOISYEXPSUM;
          } else {
            _mode = CPT;
          }
        }
        if (_mode == CPT) {

          // compute factors
          _factors = std::vector<int>(_listOfParents.size());
          int factor = 1;
          for (int j=_listOfParents.size()-1; j>=0; j--) {
            _factors.at(j) = factor;
            // LOG(INFO) << std::to_string(factor);
            factor *= allInfo[_listOfParents.at(j)]["values"].size();
          }
          _conditionalProbabilityDistributions = std::vector<std::discrete_distribution<int>>(factor);
          _conditionalProbabilities = std::vector<std::vector<double>>(factor);

          YAML::Node conditionalProbabilityTable = info["CPT"];
          for (YAML::const_iterator it = conditionalProbabilityTable.begin(); it != conditionalProbabilityTable.end(); ++it){
            std::vector<int> conditionalIndices = it->first.as<std::vector<int>>();
            std::vector<double> probabilities = it->second.as<std::vector<double>>();
            int index = 0;
            for (int i=0; i<=_factors.size()-1; i++) {
              index += _factors.at(i) * conditionalIndices.at(i);
            }
            _conditionalProbabilities.at(index) = probabilities;
            _conditionalProbabilityDistributions.at(index) = std::discrete_distribution<int>(probabilities.begin(), probabilities.end());
          }
        } else if (_mode == EXPSUM) {
          _expSumBase = info["EXPSUM"]["base"].as<int>();
        } else if (_mode == NOISYEXPSUM) {
          _expSumBase = info["NOISYEXPSUM"]["base"].as<int>();
          _noise = info["NOISYEXPSUM"]["noise"].as<float>();
        }
      }

      if (name[0] == 'x' && StringUtils::lastBitIsPrime(name) == false) {
        this->_isStateVariable = true;
      } 

      if (info["initial_dist"].IsDefined() == true) {
        auto initialProbabilities = info["initial_dist"].as<std::vector<float>>();
        _initialDist = std::discrete_distribution<int>(initialProbabilities.begin(), initialProbabilities.end());
      }

      

    }

    ~TwoStageDynamicBayesianNetworkVariable() {

    }

    // since the variables at the same stage are independent of each other, their joint entropy is the sum of individual entropies
    float computeEntropy(const std::map<std::string, int> &inputs) const {
      const std::vector<double> &probabilities = _getConditionalProbabilities(inputs);
      // compute entropy
      double entropy = 0.0;
      for (const double &element: probabilities) {
        if (element > 0) {
          entropy += - element * std::log(element);
        }
      }
      VLOG(5) << "probabilities: " << PrintUtils::vectorToString(probabilities);
      VLOG(5) << "entropy: " << std::to_string(entropy);
      return (float) entropy;
    }

    // get method
    const std::vector<double>& getProbabilities(const std::map<std::string, int> &inputs) const {
      return _getConditionalProbabilities(inputs);
    }

    // get method
    const std::vector<std::string>& getListOfParents() const {
      return _listOfParents;
    }

    // get method
    int getNumberOfInputs() const {
      return _listOfParents.size();
    }

    // get method
    int getNumberOfValues() const {
      return _numberOfValues;
    }

    // sample method
    int sampleInitialValue() const {
      return _initialDist(RandomUtils::getRandomEngine());
    }

    // sample method
    int sample(const std::map<std::string, int> &inputs) const {
      int index;
      if (_mode == CPT) {
        // VLOG(5) << PrintUtils::vectorToString(inputs);
        std::discrete_distribution<int> &distribution = _getConditionalDistribution(inputs);
        index = distribution(RandomUtils::getRandomEngine());
      } else if (_mode == SUM) {
        index = 0;
        for (const std::string &parentName: _listOfParents) {
          index += inputs.at(parentName);
        }
      } else if (_mode == EXPSUM) {
        index = 0;
        for (int i=0; i<=_listOfParents.size()-1; i++) {
          index += std::pow(_expSumBase, i) * inputs.at(_listOfParents.at(i));
        } 
      } else if (_mode == NOISYEXPSUM) {
        index = 0;
        int v;
        for (int i=0; i<=_listOfParents.size()-1; i++) {
          float r = 1.0 * std::experimental::randint(0, 9) / 10;
          if (r < _noise) {
            v = 1 - inputs.at(_listOfParents.at(i));
          } else {
            v = inputs.at(_listOfParents.at(i));
          }
          index += std::pow(_expSumBase, i) * v;
        }
      } 
      return index;
    }

    float getEntropy(const std::map<std::string, int> &inputs) const {
      // for now only support CPT
      assert(_mode == CPT);
      float entropy = 0.0;
      if (_mode == CPT) {
        for (auto &prob: _getConditionalProbabilities(inputs)) {
          if (prob > 0) {
            entropy -= prob * std::log(prob);
          }         
        };
      } else {
        LOG(FATAL) << "exact entropy computation is not supported for mode " << std::to_string(_mode);
      }
      // VLOG(5) << name;
      // VLOG(5) << "Entropy: " << std::to_string(entropy);
      assert(entropy >= 0.0);
      return entropy;
    }

    // sample method
    int sampleUniformly() const {
      return std::experimental::randint(0, _numberOfValues-1);
    }

    // get method
    float getValueFromIndex(int index) const {
      if (_listOfValues.size() != 0) {
        return _listOfValues[index];
      } else {
        return index;
      }
    }

    // info method: check whether the last digit is '
    bool isFromPreviousStage() const {
      LOG(INFO) << name.back();
      return name.back() != 39;
    }
    
    // info method
    bool isStateVariable() const {
      return _isStateVariable;
    }

  private:
    
    std::vector<float> _listOfValues; // should be copyable for sure

    bool _isStateVariable = false; // copyable for sure
    int _numberOfValues; // copyable for sure
    int _expSumBase; // copyable for sure
    float _noise; // copyable for sure
    int _mode; // copyable for sure
    
    mutable std::vector<std::discrete_distribution<int>> _conditionalProbabilityDistributions;
    mutable std::vector<std::vector<double>> _conditionalProbabilities;
    mutable std::discrete_distribution<int> _initialDist;
    // this cannot be made a const reference because we need to be able to modify the random engine when using it for sampling
    std::vector<std::string> _listOfParents;

    std::vector<int> _factors;

    int _getIndex(const std::map<std::string, int> &state) const {
      int index = 0;
      int size = _listOfParents.size();
      for (int i=0; i<=size-1; i++) {
        index += _factors.at(i) * state.at(_listOfParents.at(i));
      }
      return index;
    }

    std::discrete_distribution<int> &_getConditionalDistribution(const std::map<std::string, int> &state) const {
      return _conditionalProbabilityDistributions.at(_getIndex(state));
    }

    const std::vector<double> &_getConditionalProbabilities(const std::map<std::string, int> &state) const {
      return _conditionalProbabilities.at(_getIndex(state));
    }
};

#endif