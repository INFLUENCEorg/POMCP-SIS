#ifndef ENVIRONMENT_HPP_
#define ENVIRONMENT_HPP_

#include <map>
#include <string>

class Domain;

// the decision making environment
class Environment {
  
  public:

    Environment(const Domain &domain);

    virtual void step(std::map<std::string, int> &action, std::map<std::string, int> &observation, std::map<std::string, float> &reward, bool &done);

    virtual void reset();

    // for now this function is only used by Episode
    virtual std::map<std::string, int> getState() const;

  private:
    const Domain &_domain;
    std::map<std::string, int> _state;
};

#endif