#include "Utils.hpp"
#include "glog/logging.h"

std::mt19937 &RandomUtils::getRandomEngine() {
  return rng;
}
void RandomUtils::initRandomEngine() {
  std::random_device rd;
  rng = std::mt19937(rd());
  LOG(INFO) << "Random number generator is initialized with random device.";
}
void RandomUtils::initRandomEngine(int seed) {
  rng = std::mt19937(seed);
  LOG(INFO) << "Random number generator is initialized with seed " << std::to_string(seed) << ".";
}