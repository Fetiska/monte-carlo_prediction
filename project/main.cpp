#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using fastUint = std::uint_fast16_t;
using fastInt = std::int_fast16_t;

constexpr fastUint x{4};
constexpr fastUint y{4};

constexpr fastUint maxX{x - 1};
constexpr fastUint statesNum{x * y};
constexpr fastUint goal{statesNum - 1};
std::array<float, statesNum> values{};
std::mt19937 generator{std::random_device{}()};

consteval fastUint calculateBoundary(fastUint probabilityReciprocal) { return generator.max() / probabilityReciprocal; }

constexpr fastUint evenBoundary{calculateBoundary(2)};
constexpr fastUint randomActionBoundary{calculateBoundary(4)};

bool canMoveRight(const fastUint currentX) { return currentX < maxX; }

bool canMoveLeft(const fastUint currentX) { return currentX > 0; }

bool randomBool(const unsigned boundary) { return generator() < boundary; }//probability = boundary / generator.max()

bool updateNextStateIfNewMaxValueNotHigher(const float newMaxValue, const float maxValue) { return newMaxValue == maxValue && randomBool(evenBoundary); }

void tryUpdateNextStateAndMaxValue(const float newNextState, float& maxValue, fastUint& nextState) {
	float newMaxValue{values[newNextState]};

	if (newMaxValue > maxValue) {
		nextState = newNextState;
		maxValue = newMaxValue;
		return;
	}

	//random state
	if (updateNextStateIfNewMaxValueNotHigher(newMaxValue, maxValue)) nextState = newNextState;
	;
}

fastUint getDownState(const fastUint currentState) { return currentState + x; }
bool canMoveDown(const fastUint downState) { return downState < goal; }

fastInt getUpState(const fastInt currentState) { return currentState - static_cast<fastInt>(x); }
bool canMoveUp(const fastInt upState) { return upState > -1; }

fastUint getLeftState(const fastUint currentState) { return currentState - 1; }

void trySetNextStateToDownOrUp(const fastUint currentState, float& maxValue, fastUint& nextState) {
	//down
	{
		fastUint downState{getDownState(currentState)};
		if (canMoveDown(downState)) tryUpdateNextStateAndMaxValue(downState, maxValue, nextState);
	}

	//up
	{
		fastInt upState{getUpState(currentState)};
		if (canMoveUp(upState)) {
			float newMaxValue{values[upState]};

			if (newMaxValue > maxValue || updateNextStateIfNewMaxValueNotHigher(newMaxValue, maxValue)) nextState = upState;
		}
	}
}

int main() {
	enum class Action : fastUint {
		right,
		down,
		left,
		up
	};

	//episodes
	{
		std::vector<fastUint> states{};
		std::vector<fastInt> rewards{};

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
		for (fastUint episode{0}; episode < 9; ++episode) {
			std::cout << 'e' << episode << '\n';

			//episode
			{
				fastUint currentState{0};
				fastUint step{0};

				while (currentState != goal) {
					//current x = current position % x size
					//current y = current position // x size
					std::cout << step << ' ' << currentState % x << ' ' << currentState / x << '\n';

					states.push_back(currentState);
					rewards.push_back(-1);
					fastUint currentX{currentState % x};

					//action
					{
						if (randomBool(randomActionBoundary)) {
							std::cout << "rand\n";

							std::vector<Action> possibleActions{};

							if (canMoveRight(currentX)) possibleActions.push_back(Action::right);
							if (canMoveLeft(currentX)) possibleActions.push_back(Action::left);

							fastUint downState{getDownState(currentState)};
							if (canMoveDown(downState)) possibleActions.push_back(Action::down);

							fastInt UpState(getUpState(currentState));
							if (canMoveUp(UpState)) possibleActions.push_back(Action::up);

							Action action{possibleActions[std::uniform_int_distribution<size_t>{0, possibleActions.size() - 1}(generator)]};

							switch (action) {
								case Action::right:
									++currentState;
									break;
								case Action::left:
									--currentState;
									break;
								case Action::down:
									currentState = downState;
									break;
								case Action::up: currentState = UpState;
							}
						} else {//greedy
							std::cout << "greedy\n";
							fastUint nextState;

							//set next state
							{
								if (canMoveRight(currentX)) {
									//right
									nextState = currentState + 1;
									float maxValue{values[nextState]};

									if (canMoveLeft(currentX)) tryUpdateNextStateAndMaxValue(getLeftState(currentState), maxValue, nextState);

									trySetNextStateToDownOrUp(currentState, maxValue, nextState);
								} else {
									//left
									nextState = getLeftState(currentState);
									float maxValue{values[nextState]};

									trySetNextStateToDownOrUp(currentState, maxValue, nextState);
								}
							}

							currentState = nextState;
						}
					}

					++step;
				}
			}

			float Return{.0f};
			for (fastUint step{0}; step < states.size(); ++step) {
				Return = rewards[step] + .999999f * Return;//factor: discount factor
				float& value{values[states[step]]};
				value += .1f * (Return - value);//factor: learning rate
			}

			fastUint state{0};
			for (fastUint yi{0}; yi < y; ++yi) {
				for (fastUint xi{0}; xi < x; ++xi) {
					std::cout << values[state] << ' ';
					++state;
				}
				std::cout << '\n';
			}
		}
	}
}