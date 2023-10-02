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

constexpr fastUint maxX{x - 1};
constexpr fastUint statesNum{x * y};
constexpr fastUint goal{statesNum - 1};
std::array<float, statesNum> values{};
std::mt19937 generator{std::random_device{}()};

consteval fastUint calculateBoundary(const fastUint probabilityReciprocal) { return generator.max() / probabilityReciprocal; }

constexpr fastUint evenBoundary{calculateBoundary(2)};
constexpr fastUint randomActionBoundary{calculateBoundary(3)};

bool randomBool(const unsigned boundary) { return generator() < boundary; }//true probability = boundary / generator.max()
bool randomEvenBool() { return randomBool(evenBoundary); }

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

fastUint getLeftState(const fastUint currentState) { return currentState - 1; }

template <typename TryMoveDown, typename TryMoveUp>
	requires std::invocable<TryMoveDown, fastUint&, fastUint> && std::invocable<TryMoveUp, fastUint&, fastInt>
void tryMoveDownAndUp(const fastUint previousState, fastUint& currentState, TryMoveDown&& tryMoveDown, TryMoveUp&& tryMoveUp) {
	fastUint downState{previousState + x};
	fastInt upState(previousState - static_cast<fastInt>(x));

	//can move down
	if (downState < goal) {
		tryMoveDown(currentState, downState);
		if (upState > -1) tryMoveUp(currentState, upState);
	} else tryMoveUp(currentState, upState);
}

void tryMoveRandomly(fastUint& currentState, const fastInt newState) {
	if (randomEvenBool()) currentState = newState;
}

template <typename TryMoveLeft, typename TryMoveDown, typename TryMoveUp>
	requires std::invocable<TryMoveLeft, fastUint&, fastUint> && std::invocable<TryMoveDown, fastUint&, fastUint> && std::invocable<TryMoveUp, fastUint&, fastInt>
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

void updateValue(float& Return, const std::vector<fastUint>& states, const fastUint step) {
	Return = -1.f + .999999f * Return;//factor: discount factor
	float& value{values[states[step]]};
	value += .1f * (Return - value);//factor: learning rate
}

int main() {
	//episodes
	{
		/*
		//1st episode
		{
			std::cout << "e1\n";

			//episode
			{
				fastUint currentState{0};
				fastUint step{0};

				while (currentState != goal) {
					std::cout << step
				}
			}

			updateValues(rewards, states);
		}
		*/

		//next episodes
		for (fastUint episode{0}; episode < 300; ++episode) {
			std::cout << 'e' << episode << '\n';

			std::vector<fastUint> states{};

			//episode
			{
				fastUint currentState{0};
				fastUint step{0};

				while (currentState != goal) {
					//current x = current position % x size
					//current y = current position // x size
					std::cout << step << ' ' << currentState % x << ' ' << currentState / x << '\n';

					states.push_back(currentState);
					fastUint currentX{currentState % x};

					//action
					{
						if (randomBool(randomActionBoundary)) {
							std::cout << "rand\n";

							takeAction(
								 currentX,
								 currentState,
								 [](fastUint& currentState, fastUint previousState) { tryMoveRandomly(currentState, getLeftState(previousState)); },
								 [](fastUint& currentState, fastUint downState) { tryMoveRandomly(currentState, downState); },
								 [](fastUint& currentState, fastUint upState) {
									 tryMoveRandomly(currentState, upState);
								 });
						} else {//greedy
							std::cout << "greedy\n";

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
						}
					}

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
	}
}