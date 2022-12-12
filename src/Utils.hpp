#ifndef TEST_H
#define TEST_H

#define PRIME_CHAR 39

#include <iostream>
#include "yaml-cpp/yaml.h"
#include <string.h>
#include <type_traits>
#include <random>
#include "glog/logging.h"
#include "torch/torch.h"

extern std::mt19937 rng;

namespace StringUtils {

  template <typename T> inline std::string toString(const T &a) {
    std::ostringstream s;
    s << a;
    return s.str();
  }

  // remove the last n digits from the string
  inline std::string removeTheLastNDigits(const std::string &str, int n) {
    if (str.size() >= n) {
      return std::string(str.begin(), str.end()-n);
    } else {
      return str;
    }
  }

  // remove the prime in the end if there is one
  inline std::string removeLastPrime(const std::string &str){
    std::string newStr;
    if (str.back() == PRIME_CHAR) {
      newStr = std::string(str.begin(), str.end()-1);
      return newStr;
    } else {
      return str;
    }
  }

  // add a prime to the end if there is none
  inline std::string addLastPrime(const std::string &str) {
    if (str[0] != 'a' && str.back() != PRIME_CHAR) {
      return str + "'";
    } else {
      return str;
    }
  }

  inline bool lastBitIsPrime(const std::string &str) {
    if (str.back() == 39) {
      return true;
    } else {
      return false;
    }
  }

}

namespace ContainerUtils {
  template <class T> std::vector<T> inline setToVector(const std::set<T> &set) {
    return std::vector<T>(set.begin(), set.end());
  }
}

namespace PrintUtils {

  // convert vector to string  
  template <typename T>  inline std::string vectorToString(const std::vector<T> &vec, const std::string &connector=", ") {
    std::string str = "";
    for (const auto &element: vec) {
      str += StringUtils::toString(element) + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());;
  }

  // convert vector tuple to string  
  template <typename T>  inline std::string vectorTupleToString(const std::vector<std::tuple<T, T>> &vec, const std::string &connector=", ") {
    std::string str = "";
    for (const std::tuple<T, T> &tuple: vec) {
      str += "(" + StringUtils::toString(std::get<0>(tuple)) + ", " + StringUtils::toString(std::get<1>(tuple)) + ")" + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());;
  }

  // convert vector to tuple string
  template <class T> inline std::string vectorToTupleString(const std::vector<T> &vec, const std::string &connector=", ") {
    return "(" + vectorToString(vec, connector) + ")";
  }

  // convert vector tuple to tuple string
  template <class T> inline std::string vectorTupleToTupleString(const std::vector<std::tuple<T, T>> &vec, const std::string &connector=", ") {
    return "(" + vectorTupleToString(vec, connector) + ")";
  }

  // convert set to string
  template <typename T> inline std::string setToString(const std::set<T> &set, const std::string &connector=", ") {
    std::string str = "";
    for (const auto &element: set) {
      str += StringUtils::toString(element) + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());;
  }

  // convert set to tuple string
  template <class T> inline std::string setToTupleString(const std::set<T> &set, const std::string &connector=", ") {
    return "(" + setToString(set, connector) + ")";
  }

  template <class T> inline std::string mapToString(const std::map<std::string, T> &map, const std::string &connector=", ") {
    std::string str = "";
    for (const auto &[key, val]: map) {
      str += key + ": " + StringUtils::toString(val) + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());
  }

  template <class T> std::string mapToTupleString(const std::map<std::string, T> &map, const std::string &connector=", ") {
    return "(" + mapToString(map, connector) + ")"; 
  }

  template <class T> inline std::string mapToString(const std::unordered_map<std::string, T> &map, const std::string &connector=", ") {
    std::string str = "";
    for (const auto &[key, val]: map) {
      str += key + ": " + StringUtils::toString(val) + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());
  }

  template <class T> std::string mapToTupleString(const std::unordered_map<std::string, T> &map, const std::string &connector=", ") {
    return "(" + mapToString(map, connector) + ")"; 
  }

}

namespace FireFighterUtils {

  inline std::string environmentStateToString(const std::vector<int> &environmentState) {
    std::string str = "";
    for (int i=0; i<=(int)environmentState.size()-1; i++) {
      str += "House " + std::to_string(i+1) + ": Level " + std::to_string(environmentState[i]) + " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }

  inline std::string jointActionToString(const std::map<std::string, int> &jointAction) {
    std::string str = "";
    for (const auto &[key, val]: jointAction) {
      int agentID = std::stoi(key);
      int houseID;
      if (val == 0) {
        houseID = agentID;
      } else {
        houseID = agentID + 1;
      }
      str += "Agent " + key + " goes to House " + std::to_string(houseID) + " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }

  inline std::string jointObservationToString(const std::map<std::string, int> &jointObservation) {
    std::string str = "";
    for (const auto &[key, val]: jointObservation) {
      str += "Agent " + key + ": ";
      if (val == true) {
        str += "Fire!";
      } else {
        str += "No Fire.";
      }
      str += " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }

  inline std::string jointRewardToString(const std::map<std::string, float> &jointReward) {
    std::string str = "";
    for (const auto &[key, val]: jointReward) {
      str += "Agent " + key + ": " + std::to_string(val) + " | "; 
    }
    str = str.substr(0, str.size()-2);
    return str;
  }

  inline std::string jointActionToString(const std::unordered_map<std::string, int> &jointAction) {
    std::string str = "";
    for (const auto &[key, val]: jointAction) {
      int agentID = std::stoi(key);
      int houseID;
      if (val == 0) {
        houseID = agentID;
      } else {
        houseID = agentID + 1;
      }
      str += "Agent " + key + " goes to House " + std::to_string(houseID) + " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }

  inline std::string jointObservationToString(const std::unordered_map<std::string, int> &jointObservation) {
    std::string str = "";
    for (const auto &[key, val]: jointObservation) {
      str += "Agent " + key + ": ";
      if (val == true) {
        str += "Fire!";
      } else {
        str += "No Fire.";
      }
      str += " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }

  inline std::string jointRewardToString(const std::unordered_map<std::string, float> &jointReward) {
    std::string str = "";
    for (const auto &[key, val]: jointReward) {
      str += "Agent " + key + ": " + std::to_string(val) + " | "; 
    }
    str = str.substr(0, str.size()-2);
    return str;
  }

}

namespace GrabAChairUtils {
  inline std::string environmentStateToString(const std::vector<int> &environmentState) {
    std::string str = "";
    for (int i=0; i<=(int)environmentState.size()-1; i++) {
      str += "House " + std::to_string(i+1) + ": Level " + std::to_string(environmentState[i]) + " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }
}

namespace RandomUtils {
  std::mt19937 &getRandomEngine();
  void initRandomEngine();
  void initRandomEngine(int seed);
}

namespace StatisticsUtils {
  inline double getAverage(const YAML::Node &numbers) {
    double sum = 0.0;
    for (YAML::const_iterator it = numbers.begin(); it != numbers.end(); it++) {
      sum += it->as<double>();
    }
    return sum / numbers.size();
  }
  inline double getAverage(const std::vector<int> &numbers) {
    int size = numbers.size();
    double sum = 0.0;
    for (auto num: numbers) {
      sum += num;
    }
    return sum / size;
  }
  inline double getAverage(const std::vector<float> &numbers) {
    int size = numbers.size();
    double sum = 0.0;
    for (auto num: numbers) {
      sum += num;
    }
    return sum / size;
  }
  inline double getAverage(const std::vector<double> &numbers) {
    int size = numbers.size();
    double sum = 0.0;
    for (auto num: numbers) {
      sum += num;
    }
    return sum / size;
  }
  inline double getDiscountedReturn(const std::vector<float> &rewards, float discount_factor=1.0) {
    double sum = 0;
    double factor = 1.0;
    for (auto num: rewards) {
      sum += factor * num;
      factor *= discount_factor;
    }
    return sum;
  }
  inline double getDiscountedReturn(const std::vector<double> &rewards, float discount_factor=1.0) {
    double sum = 0;
    double factor = 1.0;
    for (auto num: rewards) {
      sum += factor * num;
      factor *= discount_factor;
    }
    return sum;
  }
}

#endif