#include "Environment.hpp"
#include "Domain.hpp"

Environment::Environment(const Domain &domain): _domain(domain){};

void Environment::step(std::map<std::string, int> &action, std::map<std::string, int> &observation, std::map<std::string, float> &reward, bool &done) {
  _domain.step(_state, action, observation, reward, done, "full");
  // update state
  std::map<std::string, int> newState = _state;
  _state.clear();
  for (auto &[key, val]: newState) {
    if (key[0] == 'x' && key.back() != 39) {
      _state[key] = val;
    }
  }
  VLOG(5) << PrintUtils::mapToTupleString(_state);
}

void Environment::reset() {
  _state = _domain.sampleInitialState();
  VLOG(1) << "Environment has been reset.";
}

std::map<std::string, int> Environment::getState() const {
  std::map<std::string, int> the_map;
  for (const auto &[key, val]: _state) {
    the_map[key] = val;
  }
  return the_map;
}