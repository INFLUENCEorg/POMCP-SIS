#ifndef SEARCH_TREE_HPP
#define SEARCH_TREE_HPP

#include <unordered_map>
#include <vector>
#include <random>

// the tree node
class TreeNode {
  public:
    int _N = 0;
    float _Q = 0.0;
    
    TreeNode() { };

    inline void update(float Return) {
      _N += 1;
      _Q = _Q + (Return - _Q) / _N;
    }
};

template <class SimulatorState>
class ActionNode;

template <class SimulatorState>
class ObservationNode: public TreeNode {
  public:

    ObservationNode(int numberOfActions): numberOfActions(numberOfActions) {
      for (int i=0; i<=numberOfActions-1; i++) {
        childrenNodes[i] = new ActionNode<SimulatorState>();
        actionsThatHaveNotBeenTaken.push(i);
      }
      root = false;
    }

    ~ObservationNode() {
      // recursively deallocate memory for the nodes
      for (auto &[key, val]: childrenNodes){
        delete val;
      }
    }

    SimulatorState sampleParticle() {
      int index = std::experimental::randint(0, (int)particles.size()-1);
      return particles.at(index);
    }

    std::unordered_map<int, ActionNode<SimulatorState>*> childrenNodes;
    std::vector<SimulatorState> particles;
    std::queue<int> actionsThatHaveNotBeenTaken;

    int numberOfActions;

    ObservationNode<SimulatorState> *getNextObservationNode(int action, int observation) {
      ObservationNode<SimulatorState> *nextObservationNodePtr = nullptr;
      ActionNode<SimulatorState> *nextActionNodePtr= this->childrenNodes.at(action);
      if (nextActionNodePtr->childrenNodes.find(observation) != nextActionNodePtr->childrenNodes.end()) {
        nextObservationNodePtr = nextActionNodePtr->childrenNodes.at(observation);
      }
      return nextObservationNodePtr;
    }

    void addNextObservationNode(int action, int observation) {
      this->childrenNodes.at(action)->childrenNodes[observation] = new ObservationNode<SimulatorState>(this->numberOfActions);
    }

    bool root = false; 

    bool isRoot() {
      return root;
    }

    void setRoot(bool is) {
      root = is;
    }
};

template <class SimulatorState>
class ActionNode: public TreeNode {
  public:

    ActionNode() { }

    ~ActionNode() {
      // recursively deallocate memory for the nodes
      for (auto &[key, val]: childrenNodes){
        delete val;
      }
    }
    std::unordered_map<int, ObservationNode<SimulatorState>*> childrenNodes;
};

template <class SimulatorState>
class SearchTree {
  public:

    int numberOfActions;

    SearchTree(int numberOfActions): numberOfActions(numberOfActions) {}
    
    // clear the database completely
    void reset() {
      if (_rootObservationNodePtr != nullptr) {
        delete _rootObservationNodePtr;
      }
      _rootObservationNodePtr = new ObservationNode<SimulatorState>(numberOfActions);
      _rootObservationNodePtr->setRoot(true);
    }

    void reset(ObservationNode<SimulatorState>* newRootObservationNodePtr) {
      if (_rootObservationNodePtr != nullptr) {
        _rootObservationNodePtr->setRoot(false);
        delete _rootObservationNodePtr;
      }
      _rootObservationNodePtr = newRootObservationNodePtr;
      _rootObservationNodePtr->setRoot(true);
    }

    ObservationNode<SimulatorState>* pop(int action, int observation) {
      if (
        _rootObservationNodePtr->childrenNodes.find(action) != _rootObservationNodePtr->childrenNodes.end() && _rootObservationNodePtr->childrenNodes.at(action)->childrenNodes.find(observation) !=  _rootObservationNodePtr->childrenNodes.at(action)->childrenNodes.end()) {
        ObservationNode<SimulatorState>* extractedNode = _rootObservationNodePtr->childrenNodes.at(action)->childrenNodes.at(observation);
        _rootObservationNodePtr->childrenNodes.at(action)->childrenNodes.erase(observation);
        return extractedNode;
      } else {
        return new ObservationNode<SimulatorState>(numberOfActions);
      }
    }

    inline bool particleDepleted() {
      return _rootObservationNodePtr->particles.empty();
    }

    ObservationNode<SimulatorState>* _rootObservationNodePtr = nullptr;
};

#endif