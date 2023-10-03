#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

using fastUint = std::uint_fast16_t;
using fastInt = std::int_fast16_t;

constexpr fastUint x{5};
constexpr fastUint y{8};

static_assert(x > 1 && y > 1);//why: more optimized random action calculation when x and y > 1

constexpr fastUint decayEpisodes{150};//number of steps to achieve min probability

constexpr fastUint maxX{x - 1};
constexpr fastUint statesNum{x * y};
constexpr fastUint goal{statesNum - 1};
std::array<float, statesNum> values{};
std::mt19937 generator{std::random_device{}()};
const float decayUpdateFactor{pow(0.1f /*min probability*/, 1.f / decayEpisodes)};
float randomActionProbability{decayUpdateFactor};

constexpr fastUint calculateBoundary(const float probability) { return generator.max() * probability; }

constexpr fastUint evenBoundary{calculateBoundary(.5f)};

bool randomBool(const unsigned boundary) { return generator() < boundary; }//true probability = boundary / generator.max()
bool randomEvenBool() { return randomBool(evenBoundary); }

template <typename invocable>
concept invocableWithCurrentState = std::invocable<invocable, fastUint&>;

template <invocableWithCurrentState Random, invocableWithCurrentState Greedy>
void choosePolicyAndTakeAction(fastUint& currentState, Random&& random, Greedy&& greedy) {
	if (randomBool(calculateBoundary(randomActionProbability))) {
		std::cout << "rand\n";

		random(currentState);
	} else {
		std::cout << "greedy\n";

		greedy(currentState);
	}
}

void moveDown(fastUint& currentState) { currentState += x; }

void randomFirstStep(fastUint& currentState) {
	//move right
	if (randomEvenBool()) ++currentState;
	else moveDown(currentState);
}

fastUint getLeftState(const fastUint state) { return state - 1; }
fastUint getDownState(const fastUint state) { return state + x; }

template <typename invocable>
concept invocableWithCurrentStateAndAnotherFastUint = std::invocable<invocable, fastUint&, fastUint>;

template <invocableWithCurrentStateAndAnotherFastUint TryMoveDown, invocableWithCurrentStateAndAnotherFastUint TryMoveUp>
void tryMoveDownAndUp(const fastUint previousState, fastUint& currentState, TryMoveDown&& tryMoveDown, TryMoveUp&& tryMoveUp) {
	fastUint downState{getDownState(previousState)};
	fastInt upState(previousState - static_cast<fastInt>(x));

	//can move down
	if (downState < goal) {
		tryMoveDown(currentState, downState);
		if (upState > -1) tryMoveUp(currentState, upState);
	} else tryMoveUp(currentState, upState);
}

template <invocableWithCurrentStateAndAnotherFastUint TryMoveLeft, invocableWithCurrentStateAndAnotherFastUint TryMoveDown, invocableWithCurrentStateAndAnotherFastUint TryMoveUp>
void takeAction(const fastUint currentX, fastUint& currentState, TryMoveLeft&& tryMoveLeft, TryMoveDown&& tryMoveDown, TryMoveUp&& tryMoveUp) {
	//can move right
	if (currentX < maxX) {
		fastUint previousState{currentState};
		++currentState;

		//can move left
		if (currentX > 0) tryMoveLeft(currentState, previousState);

		tryMoveDownAndUp(previousState, currentState, tryMoveDown, tryMoveUp);
	} else {
		fastUint previousState{currentState};
		--currentState;

		tryMoveDownAndUp(previousState, currentState, tryMoveDown, tryMoveUp);
	}
}

void tryMoveRandomly(fastUint& currentState, const fastInt newState) {
	if (randomEvenBool()) currentState = newState;
}

void randomCommonStep(fastUint& currentState, fastUint currentX) {
	takeAction(
		 currentX,
		 currentState,
		 [](fastUint& currentState, fastUint previousState) { tryMoveRandomly(currentState, getLeftState(previousState)); },
		 tryMoveRandomly,
		 tryMoveRandomly);
}

bool updateCurrentStateIfNewMaxValueNotHigher(const float newMaxValue, const float maxValue) { return newMaxValue == maxValue && randomEvenBool(); }

void tryUpdateCurrentStateAndMaxValue(const float newState, float& maxValue, fastUint& currentState) {
	float newMaxValue{values[newState]};

	if (newMaxValue > maxValue) {
		currentState = newState;
		maxValue = newMaxValue;
		return;
	}

	//random
	if (updateCurrentStateIfNewMaxValueNotHigher(newMaxValue, maxValue)) currentState = newState;
}

void updateValue(float& Return, const std::vector<fastUint>& states, const fastUint step) {
	Return = -1.f + .999999f * Return;//factor: discount factor
	float& value{values[states[step]]};
	value += .1f * (Return - value);//factor: learning rate
}

template <invocableWithCurrentState FirstStep, invocableWithCurrentStateAndAnotherFastUint CommonStep>
void executeEpisode(FirstStep firstStep, CommonStep commonStep) {
	std::vector<fastUint> states{0};

	//execute
	{
		fastUint currentState{0};

		firstStep(currentState);

		fastUint step{1};

		while (currentState != goal) {
			//current x = current position % x size
			//current y = current position // x size
			std::cout << step << ' ' << currentState % x << ' ' << currentState / x << '\n';

			states.push_back(currentState);
			fastUint currentX{currentState % x};

			commonStep(currentState, currentX);

			++step;
		}
	}

	//update values
	{
		float Return{.0f};
		for (fastUint step{static_cast<fastUint>(states.size() - 1)}; step > 0; --step) updateValue(Return, states, step);
		updateValue(Return, states, 0);
	}

	fastUint state{0};
	for (fastUint yi{0}; yi < y; ++yi) {
		for (fastUint xi{0}; xi < x; ++xi) {
			std::cout << std::setw(8) << values[state] << '|';
			++state;
		}
		std::cout << '\n';
	}
}

void executeNonFirstEpisode() {
	executeEpisode(
		 [](fastUint& currentState) {
			 choosePolicyAndTakeAction(
				  currentState,
				  randomFirstStep,
				  [](fastUint& currentState) {
					  //right better than down
					  if (values[currentState + 1] > values[getDownState(currentState)]) ++currentState;
					  else moveDown(currentState);
				  });
		 },
		 [](fastUint& currentState, fastUint currentX) {
			 choosePolicyAndTakeAction(
				  currentState,
				  [currentX](fastUint& currentState) { randomCommonStep(currentState, currentX); },
				  [currentX](fastUint& currentState) {
					  float maxValue{values[currentState]};

					  takeAction(
							currentX,
							currentState,
							[&maxValue](fastUint& currentState, const fastUint previousState) { tryUpdateCurrentStateAndMaxValue(getLeftState(previousState), maxValue, currentState); },
							[&maxValue](fastUint& currentState, const fastUint downState) { tryUpdateCurrentStateAndMaxValue(downState, maxValue, currentState); },
							[&maxValue](fastUint& currentState, const fastUint upState) {
								float newMaxValue{values[upState]};
								if (newMaxValue > maxValue || updateCurrentStateIfNewMaxValueNotHigher(newMaxValue, maxValue)) currentState = upState;
							});
				  });
		 });
}

int main() {
	//1st episode
	executeEpisode(randomFirstStep, randomCommonStep);

	//2nd episode. why separate: don't need to update random action probability in 2nd episode
	executeNonFirstEpisode();

	constexpr fastUint noDecayEpisodes{150};

	//episodes with random action probability decay
	for (fastUint episode{0}; episode < decayEpisodes; ++episode) {
		randomActionProbability *= decayUpdateFactor;
		std::cout << 'e' << episode + 2 << ' ' << randomActionProbability << '\n';

		executeNonFirstEpisode();
	}

	//episodes without random action probability decay
	for (fastUint episode{0}; episode < noDecayEpisodes; ++episode) {
		std::cout << 'e' << episode + decayEpisodes << ' ' << randomActionProbability << '\n';

		executeNonFirstEpisode();
	}
}