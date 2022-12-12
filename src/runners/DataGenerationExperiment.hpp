#ifndef DATA_GENERATION_EXPERIMENT_HPP_
#define DATA_GENERATION_EXPERIMENT_HPP_

// data generation for influence predictor training

#include "Experiment.hpp"
#include <torch/torch.h>
#include <iostream>
#include "Utils.hpp"
#include "agents/PlanningAgentSimulator.hpp"

class DataGenerationExperiment: public Experiment {

  public:

    DataGenerationExperiment(std::string pathToConfigurationFile, std::string pathToResultsFolder):Experiment(pathToConfigurationFile, pathToResultsFolder){};

    bool run(){
      
      auto intTensorOptions = torch::TensorOptions().dtype(torch::kInt32);

      // read configurations
      std::string domainName = parameters["General"]["domain"].as<std::string>();
      int horizon = parameters["General"]["horizon"].as<int>();
      std::string agentID = parameters["General"]["IDOfAgentToControl"].as<std::string>();
      int numOfRepeats = parameters["AgentComponent"][agentID]["Simulator"]["InfluencePredictor"]["numberOfSampledEpisodesForTraining"].as<int>();

      // set up the domain
      Domain *domainPtr = makeDomain(domainName);
      int numberOfActions = domainPtr->getListOfAgentActions().at(agentID);

      // set up the global simulator
      GlobalSimulator globalSimulator = GlobalSimulator(
        agentID,
        domainPtr,
        parameters["AgentComponent"]
      );

      // create placeholder tensors for inputs and outputs
      std::vector<std::string> localStates;
      std::vector<std::string> influenceSourceStates;
      std::vector<std::string> influenceDestinationStates;
      std::vector<std::string> DSeparationVariablesPerStep;
      domainPtr->constructLocalModel(agentID, localStates, influenceSourceStates, influenceDestinationStates, DSeparationVariablesPerStep);
      
      int sizeOfInputs = DSeparationVariablesPerStep.size();
      int sizeOfOutputs = influenceSourceStates.size();
      auto inputs = torch::zeros({numOfRepeats, horizon-1, sizeOfInputs}, intTensorOptions);
      auto outputs = torch::zeros({numOfRepeats, horizon-1, sizeOfOutputs}, intTensorOptions);

      LOG(INFO) << "[Influence Predictor Training] inputs: " << PrintUtils::vectorToTupleString(DSeparationVariablesPerStep);
      LOG(INFO) << "[Influence Predictor Training] size of inputs: " << inputs.sizes();
      LOG(INFO) << "[Influence Predictor Training] outputs: " << PrintUtils::vectorToTupleString(influenceSourceStates);
      LOG(INFO) << "[Influence Predictor Training] size of outputs: " << outputs.sizes();

      // data collection
      int observation;
      float reward;
      bool done;
      for (int i=0; i<=numOfRepeats-1; i++) {
        // sample one state
        auto state = globalSimulator.sampleInitialState();
        // do the trajectory simulation
        for (int step=0; step<=horizon-1; step++) {
          int action = std::experimental::randint(0, numberOfActions-1);
          globalSimulator.step(state, action, observation, reward, done);
          if (step <= horizon-2) {
            // extract local states and actions and influence sources
            int count = 0;
            for (const auto &varName: DSeparationVariablesPerStep) {
              if (varName[0] == 'a') {
                inputs[i][step][count] = action;
              } else {
                inputs[i][step][count] = state.environmentState.at(varName);
              }
              count += 1;
            }
            // outputs
            for (int j=0; j<=influenceSourceStates.size()-1; j++) {
              if (influenceSourceStates.at(j)[0] != 'a') {
                outputs[i][step][j] = state.environmentState[influenceSourceStates[j]];
              } else {
                if (step != 0) {
                  outputs[i][step-1][j] = state.environmentState[influenceSourceStates[j]];
                }
              }
            }
          } else {
            for (int j=0; j<=influenceSourceStates.size()-1; j++) {
              if (influenceSourceStates.at(j)[0] == 'a') {
                outputs[i][step-1][j] = state.environmentState[influenceSourceStates[j]];
              }
            }
          }
        }
      }

      // save data
      LOG(INFO) << "inputs path: " << pathToResultsFolder+"/inputs.pt";
      LOG(INFO) << "outputs path: " << pathToResultsFolder+"/outputs.pt";
      torch::save(inputs, pathToResultsFolder+"/inputs.pt");
      torch::save(outputs, pathToResultsFolder+"/outputs.pt");

      delete domainPtr;

      return true;
  }

};

#endif