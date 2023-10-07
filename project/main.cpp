#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

using fastUint = std::uint_fast16_t;
using fastInt = std::int_fast16_t;

fastUint xSize;
fastUint ySize;
fastUint decayEpisodes;//number of steps to achieve min probability
float minRandomActionProbability;
float learningRate;
float discount;

fastUint maxX;
fastUint statesNum;
fastUint goal;
std::vector<float> values;
std::mt19937 generator{std::random_device{}()};
float randomActionProbability;

constexpr fastUint calculateBoundary(const float probability) { return generator.max() * probability; }

constexpr fastUint evenBoundary{calculateBoundary(.5f)};

bool randomBool(const unsigned boundary) { return generator() < boundary; }//true probability = boundary / generator.max()
bool randomEvenBool() { return randomBool(evenBoundary); }

template <typename invocable>
concept invocableWithCurrentState = std::invocable<invocable, fastUint&>;

template <invocableWithCurrentState Random, invocableWithCurrentState Greedy>
void choosePolicyAndTakeAction(fastUint& currentState, Random&& random, Greedy&& greedy) {
	if (randomBool(calculateBoundary(randomActionProbability))) random(currentState);
	else greedy(currentState);
}

void moveDown(fastUint& currentState) { currentState += xSize; }

void randomFirstStep(fastUint& currentState) {
	//move right
	if (randomEvenBool()) ++currentState;
	else moveDown(currentState);
}

fastUint getLeftState(const fastUint state) { return state - 1; }
fastUint getDownState(const fastUint state) { return state + xSize; }

template <typename invocable>
concept invocableWithCurrentStateAndAnotherFastUint = std::invocable<invocable, fastUint&, fastUint>;

template <invocableWithCurrentStateAndAnotherFastUint TryMoveDown, invocableWithCurrentStateAndAnotherFastUint TryMoveUp>
void tryMoveDownAndUp(const fastUint previousState, fastUint& currentState, TryMoveDown&& tryMoveDown, TryMoveUp&& tryMoveUp) {
	fastUint downState{getDownState(previousState)};
	fastInt upState{static_cast<fastInt>(previousState) - static_cast<fastInt>(xSize)};

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
	Return = -1.f + discount * Return;
	float& value{values[states[step]]};
	value += learningRate * (Return - value);
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
			states.push_back(currentState);
			fastUint currentX{currentState % xSize};

			commonStep(currentState, currentX);

			++step;
		}

		std::cout << step << '\n';
	}

	//update values
	{
		float Return{.0f};
		for (fastUint step{static_cast<fastUint>(states.size() - 1)}; step > 0; --step) updateValue(Return, states, step);
		updateValue(Return, states, 0);
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

void run() {
}

int main() {
	do {
		fastUint episodes;

		//input
		while (true) {
			std::cout << "enter\n x size\n y size\n total number of episodes\n learning rate\n discount factor\n number of episodes with decay of exploration probability\n exploration probability after decay\n";

			int xSizeInput, ySizeInput, episodesInput, decayEpisodesInput;

			std::cin >> xSizeInput >> ySizeInput >> episodesInput >> learningRate >> discount >> decayEpisodesInput >> minRandomActionProbability;

			std::cout << '\n';

			bool valid{true};

			if (xSizeInput < 2) {
				std::cout << "x size must be > 1\n";
				valid = false;
			}
			if (ySizeInput < 2) {
				std::cout << "y size must be > 1\n";
				valid = false;
			}
			if (episodesInput < 1) {
				std::cout << "total episodes must be > 0\n";
				valid = false;
			}
			if (learningRate <= .0f) {
				std::cout << "learning rate must be > 0\n";
				valid = false;
			}
			if (discount < .0f || discount > 1.f) {
				std::cout << "discount factor must be between 0 and 1\n";
				valid = false;
			}
			if (decayEpisodesInput < 0) {
				std::cout << "number of episodes with decay of exploration probability be >= 0\n";
				valid = false;
			}
			if (minRandomActionProbability < .0f || minRandomActionProbability > 1.f) {
				std::cout << "exploration probability after decay must be between 0 and 1\n";
				valid = false;
			}
			if (std::cin.fail()) {
				std::cout << xSizeInput;
				std::cout << "use valid numbers\n";
				std::cin.clear();
				std::cin.ignore();
				valid = false;
			}

			if (valid) {
				xSize = xSizeInput;
				ySize = ySizeInput;
				episodes = episodesInput;
				decayEpisodes = decayEpisodesInput;
				break;
			};

			std::cout << '\n';
		}

		//initialize variables
		maxX = xSize - 1;
		statesNum = xSize * ySize;
		goal = statesNum - 1;
		values.resize(statesNum, .0f);

		//1st episode
		std::cout << "1 ";
		executeEpisode(randomFirstStep, randomCommonStep);

		const float decayUpdateFactor{pow(minRandomActionProbability, 1.f / decayEpisodes)};
		randomActionProbability = decayUpdateFactor;

		//2nd episode. why separate: don't need to update random action probability in 2nd episode
		std::cout << "2 ";
		executeNonFirstEpisode();

		const fastUint firstNonDecayEpisode{decayEpisodes + 1};

		//episodes with random action probability decay
		for (fastUint episode{3}; episode < firstNonDecayEpisode; ++episode) {
			randomActionProbability *= decayUpdateFactor;
			std::cout << episode << ' ';

			executeNonFirstEpisode();
		}

		//episodes without random action probability decay
		for (fastUint episode{firstNonDecayEpisode}; episode < episodes + 1; ++episode) {
			std::cout << episode << ' ';

			executeNonFirstEpisode();
		}

		//print values
		fastUint state{0};
		for (fastUint y{0}; y < ySize; ++y) {
			for (fastUint x{0}; x < xSize; ++x) {
				std::cout << std::setw(8) << values[state] << '|';
				++state;
			}
			std::cout << '\n';
		}

		std::cout << "press 'enter' to run program again\n";
		std::cin.ignore();
		if (std::cin.get() != '\n') break;
	} while (true);
}