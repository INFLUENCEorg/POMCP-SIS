#ifndef EXPERIMENT_HPP_
#define EXPERIMENT_HPP_

#include "agents/AgentComponent.hpp"
#include "agents/AtomicAgent.hpp"
#include "agents/PlanningAgent.hpp"

#include <fstream>
#include <map>
#include <any>
#include "yaml-cpp/yaml.h"
#include "domains/Domain.hpp"
#include "domains/GridTraffic/GridTrafficDomain.hpp"
#include "domains/GrabAChair/GrabAChairDomain.hpp"
#include "domains/FireFighter/FireFighterDomain.hpp"
#include "domains/InfluenceCycle/InfluenceCycleDomain.hpp"
#include <time.h>
#include <random>

// the experiment
class Experiment {

public:

    Experiment(const std::string &pathToConfigurationFile, const std::string &pathToResultsFolder):pathToResultsFolder(pathToResultsFolder){
      parameters = YAML::LoadFile(pathToConfigurationFile);
      LOG(INFO) << "\n-------------Experimental Parameters:-------------\n" << parameters << "\n--------------------------------------------------\n";
      // set up random seed
      if (parameters["Experiments"]["seed"].IsDefined() == true) {
        int seed = parameters["Experiments"]["seed"].as<int>();
        RandomUtils::initRandomEngine(seed);
      } else {
        RandomUtils::initRandomEngine();
      }
    }

    virtual bool run(){
        std::cerr << "Error: Experiment Not Implemented." << std::endl;
        return false;
    }

    Domain *makeDomain(const std::string &domain) {
      if (domain == "GridTraffic") {
        return new GridTrafficDomain(parameters);
      } else if (domain == "GrabAChair") {
        return new GrabAChairDomain(parameters);
      } else if (domain == "FireFighter") {
        return new FireFighterDomain(parameters);
      } else if (domain == "InfluenceCycle") 
        return new InfluenceCycleDomain(parameters);      
      else {
        std::string message = "domain " + domain + " is not supported.";
        throw std::invalid_argument(message);
      }
    }

protected:
    std::string pathToResultsFolder;
    YAML::Node parameters;
};

class TestingExperiment: public Experiment {

public:

    TestingExperiment(const std::string &pathToConfigurationFile, const std::string &pathToResultsFolder):Experiment(pathToConfigurationFile, pathToResultsFolder){};

    bool run(){
        LOG(INFO) << "Testing Experiment Finished.";
        return true;
    }

};

class PlanningExperiment: public Experiment {

  public:

    PlanningExperiment(const std::string &pathToConfigurationFile, const std::string &pathToResultsFolder):Experiment(pathToConfigurationFile, pathToResultsFolder){};

    bool run(){
      bool fullLogging = parameters["Experiment"]["fullLogging"].as<bool>();
      std::string IDOfAgentToControl = parameters["General"]["IDOfAgentToControl"].as<std::string>();
      
      // create the csv file for experimental results
      YAML::Node resultsYAML;
      int numOfRepeats = parameters["Experiment"]["repeat"].as<int>();
      
      // build up the domain
      std::string domainName = parameters["General"]["domain"].as<std::string>();
      LOG(INFO) << "Domain Name: " << domainName;
      Domain *domainPtr = makeDomain(domainName);

      // load parameters for episodes
      int horizon = parameters["General"]["horizon"].as<int>();
      float discountFactor = parameters["General"]["discountFactor"].as<float>();

      // build up the environment
      Environment environment = domainPtr->makeEnvironment();
      VLOG(1) << "Environment has been set up.";

      bool saveReplay = parameters["Experiment"]["saveReplay"].as<bool>();
      std::string pathToReplaysFolder = pathToResultsFolder+"/replays/";

      // build up the agents
      AgentComponent* agentComponentPtr = domainPtr->makeAgentComponent();

      // to store moving average of episodic returns
      std::vector<float> movingAverages;
      std::vector<float> undiscountedMovingAverages;

      for (int episodeID=0; episodeID<=numOfRepeats-1; episodeID++){
        
        // prepare for replay
        YAML::Node replay;
        
        std::string episodeIDStr = std::to_string(episodeID);
        YAML::Node episodeResults = YAML::Node();
        std::string prefix = "[Episode " + episodeIDStr + "] ";
        
        ////////////////////////////////////// EPISODE /////////////////////////////////////
        
        // information - beginning of an episode
        LOG(INFO) << "===================================================================";
        LOG(INFO) << "Episode " << episodeIDStr << " has been dispatched.";

        // timing code for debugging
        double actTime = 0.0;
        double stepTime = 0.0;
        double observeTime = 0.0;

        // reset the environment and agents
        agentComponentPtr->reset();
        environment.reset();
        
        // interative variables
        std::map<std::string, int> action;
        std::map<std::string, int> observation;
        std::map<std::string, float> reward;
        bool done = false;

        // prepare statistics
        // returns need to be computed backwards
        std::map<std::string, std::vector<float>> rewards;

        // the interaction loop
        for (int i=0; i<=horizon-1; i++) {
          
          VLOG(3) << "---------------------------------------------------------------------------";
          VLOG(3) << "---------------------------------------------------------------------------";

          std::string stepID = std::to_string(i);
          
          if (saveReplay == true) {
            replay[stepID]["state"] = environment.getState();
          }

          // agents make decisions
          clock_t begin = std::clock();
          agentComponentPtr->act(action, episodeResults);
          actTime += (double)(std::clock()-begin)/CLOCKS_PER_SEC;
          
          // the environment transitions
          begin = std::clock();
          environment.step(action, observation, reward, done);
          stepTime += (double)(std::clock()-begin)/CLOCKS_PER_SEC;
          VLOG(3) << "[Environment] step " << std::to_string(i);
          VLOG(3) << "[Environment] reward " << std::to_string(reward.at(IDOfAgentToControl));
          
          // agents update internal states
          begin = std::clock();
          agentComponentPtr->observe(observation);
          observeTime += (double)(std::clock()-begin)/CLOCKS_PER_SEC;

          // compute discounted and undiscounted returns
          for (auto &[agentID, agentReward]: reward) {
            if (rewards.find(agentID) == rewards.end()) {
              rewards[agentID] = std::vector<float>({agentReward});
            } else {
              rewards[agentID].push_back(agentReward);
            }
          }

          // replay
          if (saveReplay == true) {
            replay[stepID]["action"] = action;
            replay[stepID]["observation"] = observation;
            replay[stepID]["reward"] = reward;
          }

          // terminate if done
          if (done == true) {
            break;
          }
        }

        agentComponentPtr->log(episodeResults);

        // save replay 
        if (saveReplay == true) {
          std::ofstream fout(pathToReplaysFolder+"episode"+episodeIDStr+".yaml");
          fout << replay;
        }

        // logging
        VLOG(2) << "[Episode] act time in total: " << std::to_string(actTime);
        VLOG(2) << "[Episode] step time in total: " << std::to_string(stepTime);
        VLOG(2) << "[Episode] observe time in total: " << std::to_string(observeTime);

        ////////////////////////////////////// EPISODE /////////////////////////////////////
        
        // logging
        for (int j=0; j<=domainPtr->getNumberOfAgents()-1; j++){
          auto &agentID = domainPtr->getListOfAgentIDs()[j];
          
          double discounted_episodic_return = StatisticsUtils::getDiscountedReturn(rewards.at(agentID), discountFactor);
          double undiscounted_episodic_return = StatisticsUtils::getDiscountedReturn(rewards.at(agentID), 1.0);

          if ((fullLogging == true) || (agentID == IDOfAgentToControl)) {
            episodeResults[agentID]["Return"] = discounted_episodic_return;
            episodeResults[agentID]["Undiscounted_return"] = undiscounted_episodic_return;
          }
          
          // message printing
          std::string returnMessage =  prefix + "Agent " + agentID + " Discounted Episodic Return: " + std::to_string(discounted_episodic_return);
          std::string undiscountedReturnMessage =  prefix + "Agent " + agentID + " Undiscounted Episodic Return: " + std::to_string(undiscounted_episodic_return);

          std::string movingAvgMessage;
          if (movingAverages.size() < domainPtr->getNumberOfAgents()) {
            movingAverages.push_back(discounted_episodic_return);
          } else {
            movingAverages[j] = (movingAverages[j] * episodeID + discounted_episodic_return) / (episodeID+1);
          }
          movingAvgMessage = prefix + "Agent " + agentID + " Moving Average of discounted returns: " + std::to_string(movingAverages[j]);

          std::string undiscountedMovingAvgMessage;
          if (undiscountedMovingAverages.size() < domainPtr->getNumberOfAgents()) {
            undiscountedMovingAverages.push_back(undiscounted_episodic_return);
          } else {
            undiscountedMovingAverages[j] = (undiscountedMovingAverages[j] * episodeID + undiscounted_episodic_return) / (episodeID+1);
          }
          undiscountedMovingAvgMessage = prefix + "Agent " + agentID + " Moving Average of undiscounted returns: " + std::to_string(undiscountedMovingAverages[j]);

          std::string timeMessage; 
          timeMessage = prefix + "Agent " + agentID + " Average Decision Making Time Per Step: ";
          double averageTime = StatisticsUtils::getAverage(episodeResults[agentID]["number_of_seconds_per_step"]);
          timeMessage += std::to_string(averageTime);

          std::string simMessage;
          simMessage = prefix + "Agent " + agentID + " Number of simulations Per Step: ";
          double avgNumSims = StatisticsUtils::getAverage(episodeResults[agentID]["number_of_simulations_per_step"]);
          simMessage += std::to_string(avgNumSims);

          std::string particleMessage;
          particleMessage = prefix + "Agent " + agentID + " Number of particles before simulation Per Step: ";
          double avgNumParticles = StatisticsUtils::getAverage(episodeResults[agentID]["number_of_particles_before_simulation"]);
          particleMessage += std::to_string(avgNumParticles);

          if (agentID == IDOfAgentToControl) {
            LOG(INFO) << returnMessage;
            LOG(INFO) << undiscountedReturnMessage;
            LOG(INFO) << undiscountedMovingAvgMessage;
            LOG(INFO) << movingAvgMessage;
            LOG(INFO) << timeMessage;
            LOG(INFO) << simMessage;
            LOG(INFO) << particleMessage;
            resultsYAML[episodeIDStr] = episodeResults;
          }   
        }
      }

      agentComponentPtr->save(pathToResultsFolder);
      
      delete domainPtr;
      delete agentComponentPtr;

      // save results
      std::ofstream resultsYAMLFile;
      resultsYAMLFile.open(pathToResultsFolder+"/results.yaml");
      resultsYAMLFile << resultsYAML;
      resultsYAMLFile.close();

      return true;
  }
};

#endif