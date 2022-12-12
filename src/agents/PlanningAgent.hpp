#ifndef PLANNING_AGENT_HPP_
#define PLANNING_AGENT_HPP_

#include "AtomicAgent.hpp"
#include "SearchTree.hpp"
#include "PlanningAgentSimulator.hpp"

// constants for simulators
#define GS 0
#define IALS 1

// constants for action selection methods
// we can use search trees independently
// or in a combined way
#define COMBINED 0 // default
#define INDEPENDENT 1

// constants for the option to use exploration bonus for simulators
#define META_EXP_BOTH 0 // default
#define META_EXP_NONE 1
#define META_EXP_IALS 2
#define META_EXP_GS 3 

#define NN 0
#define MLE 1
#define EXACT 2
#define NONE 3

class AbstractPOMCPAtomicAgent: public AtomicAgent {
  public:
    AbstractPOMCPAtomicAgent(
      const std::string &agentID, 
      int numberOfActions, 
      int numberOfStepsToPlan, 
      const YAML::Node &agentParameters 
    );

    virtual void observe(int observation) = 0;

    virtual void reset();
     
    virtual int act(YAML::Node results);

    // a helper class for the planning agents 
    class Looper {
      public:
        Looper(AbstractPOMCPAtomicAgent* planningAgentPtr): _planningAgentPtr(planningAgentPtr) {

        }
        void reset() {
          elapsedTime = 0.0;
          simulationID = 0;
        }
        void start() {
          VLOG(4) << "Simulation " << std::to_string(simulationID) << " started.";
          beginTime = std::clock();
        }
        void end() {
          simulationID += 1;
          double timeThisSimulation = double(std::clock()-beginTime)/CLOCKS_PER_SEC;
          elapsedTime += timeThisSimulation;
          VLOG(4) << "Simulation took " << std::to_string(timeThisSimulation);
        }
        void log(YAML::Node &results) {
          results["number_of_simulations_per_step"].push_back(simulationID);
          results["number_of_seconds_per_step"].push_back(elapsedTime);
          VLOG(3) << "[Looper] number of simulations at this step: " << std::to_string(simulationID);
          VLOG(3) << "[Looper] number of seconds at this step: " << std::to_string(elapsedTime);
        }
        bool finished() {
          bool finished = false;
          if (_planningAgentPtr->_numberOfSecondsPerStep > 0.0 && elapsedTime >= _planningAgentPtr->_numberOfSecondsPerStep) {
            VLOG(3) << "[Agent " + _planningAgentPtr->_agentID + "]: reached planning time.";
            finished = true;
          } else if (_planningAgentPtr->_numberOfSimulationsPerStep > 0 && simulationID >= _planningAgentPtr->_numberOfSimulationsPerStep) {
            VLOG(3) << "[Agent " + _planningAgentPtr->_agentID + "]: reached number of simulations.";
            finished = true;
          }
          return finished;
        }
      private:
        AbstractPOMCPAtomicAgent* _planningAgentPtr;
        double elapsedTime = 0.0;
        int simulationID = 0;
        std::clock_t beginTime;
    };

  protected:

    template <class A, class B>
    int selectAction(ObservationNode<A>* nodePtr1, ObservationNode<B>* nodePtr2, bool UCB) const;

    template <class A>
    int selectAction(ObservationNode<A>* nodePtr1, bool UCB) const;

    // the looper serves planning
    Looper _looper;
    virtual int plan(YAML::Node &results) = 0;

    // particle reinvigoration
    bool _particleReinvigoration = false;
    float _particleReinvigorationRate;

    // if we are going to use a separate particle filtering process
    bool _reuseSimulationsToFilterParticle;

    // simulation/rollout related
    int _numberOfSimulationsPerStep = -1;
    double _numberOfSecondsPerStep = -1.0;
    int _planningHorizon;
    int _numberOfParticles;
    float _discountFactor;
    float _discountHorizon;

    // UCB related
    float _explorationConstant;

    // the action that was taken previously
    int _previousActionTaken;

    // abstract method - if particle depleted has already occurred?
    virtual bool _particleDepleted() const = 0;

    // compute exploration bonus
    float computeExplorationBonus(int NTotal, int N, float explorationBonus) const;

    // updateSearchTree
    template <class SimulatorState, class Simulator> 
    void _updateSearchTree(SearchTree<SimulatorState> *searchTreePtr, Simulator *simulatorPtr, int action, int observation);

    // rollout
    template <class SimulatorState, class Simulator>
    float rollout(
      SimulatorState &sampledState, 
      Simulator *simulatorPtr, 
      int horizon, 
      int depth, 
      std::vector<float> *crossEntropies=nullptr, 
      std::vector<float> *entropies=nullptr);

    // simulate
    template <class SimulatorState1, class SimulatorState2, class Simulator>
    float simulate(
      SimulatorState1 &sampledState,
      Simulator *simulatorPtr,
      ObservationNode<SimulatorState1> *thisNodePtr,
      int horizon,
      int depth,
      ObservationNode<SimulatorState2> *otherNodePtr=nullptr,
      std::vector<float> *crossEntropies=nullptr,
      std::vector<float> *entropies=nullptr);

    virtual void _initializeResultsYAML(YAML::Node results) = 0;
};

// the normal POMCP agent that utilizes one simulator
template <class SimulatorState, class Simulator>
class POMCPAtomicAgent: public AbstractPOMCPAtomicAgent {
  public:
    POMCPAtomicAgent(
      const std::string &agentID, 
      int numberOfActions, 
      int numberOfStepsToPlan, 
      const YAML::Node &agentParameters, 
      Simulator* simulatorPtr
    );

    ~POMCPAtomicAgent();

    void observe(int observation) override;

    void reset() override;

  private:
    int plan(YAML::Node &results) override;

    Simulator *_simulatorPtr;
    
    // treat the search tree as a database
    SearchTree<SimulatorState> *_searchTreePtr = nullptr;

    bool _particleDepleted() const override;
    
    void _initializeResultsYAML(YAML::Node results) override;
};

// the POMCP agent that can use two simulators
class MixedPOMCPAtomicAgent: public AbstractPOMCPAtomicAgent {
  public:
    MixedPOMCPAtomicAgent(
      const std::string &agentID, 
      int numberOfActions, 
      int numberOfStepsToPlan, 
      const YAML::Node &agentParameters, 
      SelfImprovingSimulator* simulatorPtr
    );

    ~MixedPOMCPAtomicAgent();

    void observe(int observation) override;

    void reset() override;

    void log(YAML::Node agentResults) const override;

    void save(const std::string &pathToResultsFolder) const override;

    class MetaAgent {
      public:

        MetaAgent(MixedPOMCPAtomicAgent* planningAgentPtr, const YAML::Node &metaParameters): _planningAgentPtr(planningAgentPtr) {
          // meta exploration constant
          _metaExplorationConstant = metaParameters["metaExplorationConstant"].as<float>();
          // meta exploration bonus decides to which simulator we apply exploration bonus
          std::string tmp = metaParameters["metaExplorationBonus"].as<std::string>();
          if (tmp == "BOTH") {
            _metaExplorationBonus = META_EXP_BOTH;
          } else if (tmp == "GS") {
            _metaExplorationBonus = META_EXP_GS;
          } else if (tmp == "IALS") {
            _metaExplorationBonus = META_EXP_IALS;
          } else if (tmp == "NONE") {
            _metaExplorationBonus = META_EXP_NONE;
          } else {
            LOG(FATAL) << "Meta Exploration Bonus not supported.";
          }
          // lambda
          _lambda = metaParameters["lambda"].as<float>();
          // computation cost
          _computationCostByHorizon = metaParameters["computationCostByHorizon"].as<bool>();
        }

        // to be performed at the beginning of a planning step
        void resetStep() {
          // reset statistics
          countGlobal = 0;
          countIALS = 0;
          valueGlobal = 0.0;
          valueIALS = 0.0;
          lossCount = 0;
          sumCrossEntropy = 0.0;
          sumEntropy = 0.0;
          sumKLBonus = 0.0;
        }

        // to be performed at the end of a planning step
        void wrapUpStep(){
          // save statistics
          _globalCounts.push_back(countGlobal);
          _IALSCounts.push_back(countIALS);
          _globalValues.push_back(valueGlobal);
          _IALSValues.push_back(valueIALS);
          _averageKLBonuses.push_back(sumKLBonus / lossCount);
          _averageCrossEntropies.push_back(sumCrossEntropy / lossCount);
          _averageEntropies.push_back(sumEntropy / lossCount);
          VLOG(3) << "[META] global simulator count: " << std::to_string(countGlobal) << " | value: " << std::to_string(valueGlobal) << " | KL Bonus: " << std::to_string(sumKLBonus / lossCount);
          VLOG(3) << "[META] IALS count: " << std::to_string(countIALS);
        }

        void updateStep(const std::vector<float> &crossEntropies, const std::vector<float> &entropies) {
          // update the sums
          int size = crossEntropies.size();
          for (int i=0; i<=size-1; i++) {
            sumCrossEntropy += crossEntropies.at(i);
            sumEntropy += entropies.at(i);
            sumKLBonus += crossEntropies.at(i) - entropies.at(i);
            lossCount += 1;
          }
          // update the values 
          float averageKLBonuses = sumKLBonus / lossCount;
          assert(std::abs(averageKLBonuses - (sumCrossEntropy - sumEntropy)/lossCount) < 0.1);
          valueGlobal = averageKLBonuses - _computeGlobalSimulatorComputationCost(this->_planningAgentPtr->_planningHorizon);
          valueIALS = -_computeIALSComputationCost(this->_planningAgentPtr->_planningHorizon);
        }

        void log(YAML::Node &agentResults) const {
          assert(_planningAgentPtr->_planningHorizon == 0);

          agentResults["globalCounts"] = _globalCounts;
          agentResults["globalValues"] = _globalValues;
          agentResults["IALSCounts"] = _IALSCounts;
          agentResults["IALSValues"] = _IALSValues;
          agentResults["averageKLBonuses"] = _averageKLBonuses;
          agentResults["averageCrossEntropies"] = _averageCrossEntropies;
          agentResults["averageEntropies"] = _averageEntropies;

          // summarize the episode
          LOG(INFO) 
            << "[META] Global Simulator Count: " << std::to_string(StatisticsUtils::getAverage(_globalCounts)) 
            << " | Value: " << std::to_string(StatisticsUtils::getAverage(_globalValues)) 
            << " | Avg KL Bonus: " << std::to_string(StatisticsUtils::getAverage(_averageKLBonuses)) 
            << " | Avg Cross Entropy: " << std::to_string(StatisticsUtils::getAverage(_averageCrossEntropies)) 
            << " | Avg Entropy: " << std::to_string(StatisticsUtils::getAverage(_averageEntropies));
          LOG(INFO) 
            << "[META] IALS Count: " << std::to_string(StatisticsUtils::getAverage(_IALSCounts)) 
            << " | Value: " << std::to_string(StatisticsUtils::getAverage(_IALSValues));
        }

        void resetEpisode() {
          // reset
          _globalCounts.clear();
          _globalCounts.reserve(_planningAgentPtr->_numberOfStepsToPlan);
          _IALSCounts.clear();
          _IALSCounts.reserve(_planningAgentPtr->_numberOfStepsToPlan);
          _globalValues.clear();
          _globalValues.reserve(_planningAgentPtr->_numberOfStepsToPlan);
          _IALSValues.clear();
          _IALSValues.reserve(_planningAgentPtr->_numberOfStepsToPlan);
          _averageKLBonuses.clear();
          _averageKLBonuses.reserve(_planningAgentPtr->_numberOfStepsToPlan);
          _averageCrossEntropies.clear();
          _averageCrossEntropies.reserve(_planningAgentPtr->_numberOfStepsToPlan);
          _averageEntropies.clear();
          _averageEntropies.reserve(_planningAgentPtr->_numberOfStepsToPlan);
        }

        // GS refers to global simulator and IALS refers to influence-augmented local simulator
        int pickSimulator() {
          int simulatorType;
          if (_planningAgentPtr->_particleDepleted() == true) {
            LOG(FATAL) << "if particle depletion has occurred in both search trees, planning should not have happened.";
          }
          if (_planningAgentPtr->_globalSearchTreePtr->particleDepleted() == true) {
            simulatorType = IALS;
          } else if (_planningAgentPtr->_IALSSearchTreePtr->particleDepleted() == true) {
            simulatorType = GS;
          } else if (countGlobal == 0) {
            simulatorType = GS;
          } else if (countIALS == 0) {
            simulatorType = IALS;
          } else {
            // pick simulators according to meta statistics
            int countSum = countGlobal + countIALS;
            // compute meta uct exploration bonus
            float globalExplorationBonus = 0.0;
            float IALSExplorationBonus = 0.0;
            if (_metaExplorationBonus == META_EXP_BOTH || _metaExplorationBonus == META_EXP_GS) {
              globalExplorationBonus = _metaExplorationConstant * sqrtf(std::log(countSum) / countGlobal);
            }
            if (_metaExplorationBonus == META_EXP_BOTH || _metaExplorationBonus == META_EXP_IALS) {
              IALSExplorationBonus = _metaExplorationConstant * sqrtf(std::log(countSum) / countIALS);
            }
            if (valueGlobal + globalExplorationBonus >= valueIALS + IALSExplorationBonus) {
              simulatorType = GS;
            } else {
              simulatorType = IALS;
            }
          }
          if (simulatorType == GS) {
            countGlobal += 1;
          } else if (simulatorType == IALS) {
            countIALS += 1;
          }
          return simulatorType;
        }

      private:
        
        float _computeGlobalSimulatorComputationCost(int planningHorizon) {
          if (_computationCostByHorizon == true) {
            return _lambda * planningHorizon / _planningAgentPtr->_numberOfStepsToPlan;
          } else {
            return _lambda;
          }
        }

        float _computeIALSComputationCost(int planningHorizon) {
          return 0.0;
        }

        // pointer to the planningAgent
        MixedPOMCPAtomicAgent* _planningAgentPtr;

        // hyperparameters
        int _metaExplorationBonus;
        float _metaExplorationConstant;
        float _lambda; 
        bool _computationCostByHorizon = false;
        
        // for control
        int countGlobal;
        int countIALS;
        float valueIALS;
        float valueGlobal;
        int lossCount; 
        float sumCrossEntropy;
        float sumEntropy;
        float sumKLBonus;

        // we cannot send statistics to results at every time step because I don't know
        // vector = episode
        std::vector<int> _globalCounts;
        std::vector<int> _IALSCounts;
        std::vector<float> _globalValues;
        std::vector<float> _IALSValues;
        std::vector<float> _averageKLBonuses;
        std::vector<float> _averageCrossEntropies;
        std::vector<float> _averageEntropies;
    };

    // meta parameters
    bool _computeRolloutLoss;
    bool _storeRolloutData;
    bool _entropyEstimation;
    int _entropyEstimationType;

  private:
    
    // the simulator
    SelfImprovingSimulator *_simulatorPtr;
    
    // the databases
    SearchTree<SelfImprovingSimulator::IndexedGlobalSimulatorState>* _globalSearchTreePtr;
    SearchTree<InfluenceAugmentedLocalSimulator::InfluenceAugmentedLocalSimulatorState>* _IALSSearchTreePtr; 

    // action selection method during the tree search: do we look at the statistics of two trees or a single tree?
    int _actionSelectionMethod;

    MetaAgent metaAgent;

    bool _particleDepleted() const override;

    int plan(YAML::Node &results) override;

    void _initializeResultsYAML(YAML::Node results) override;

    bool _saveReplayBuffer;
};


#endif