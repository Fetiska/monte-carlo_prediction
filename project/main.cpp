#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using fastUint = std::uint_fast16_t;
using fastInt = std::int_fast16_t;

constexpr fastUint x{9};
constexpr fastUint y{9};

constexpr fastUint maxX{x - 1};
constexpr fastUint statesNum{x * y};
constexpr fastUint goal{statesNum - 1};
std::array<float, statesNum> values{};

fastUint getDownState(const fastUint currentState) { return currentState + x; }
bool canMoveDown(const fastUint downState) { return downState < goal; }

void tryUpdateNextStateAndMaxValue(const float newNextState, float& maxValue, fastUint& nextState) {
	float newMaxValue{values[newNextState]};

	if (newMaxValue > maxValue) {
		nextState = newNextState;
		maxValue = newMaxValue;
	}
}

fastInt getUpState(const fastInt currentState) { return currentState - static_cast<fastInt>(x); }
bool canMoveUp(const fastInt upState) { return upState > -1; }

void trySetNextStateToDownOrUp(const fastUint currentState, float& maxValue, fastUint& nextState) {
	//down
	{
		fastUint downState{getDownState(currentState)};
		if (canMoveDown(downState)) tryUpdateNextStateAndMaxValue(downState, maxValue, nextState);
	}

	//up
	{
		fastInt upState{getUpState(currentState)};
		if (canMoveUp(upState)) nextState = upState;
	}
}

bool canMoveRight(const fastUint currentX) { return currentX < maxX; }

void setMaxValue(fastUint nextState, float& maxValue) { maxValue = values[nextState]; }

fastUint getLeftState(const fastUint currentState) { return currentState - 1; }

int main() {
	std::mt19937 generator{std::random_device{}()};
	std::uniform_real_distribution<float> policyDistribution{0, 1};

	enum class Action : fastUint {
		right,
		down,
		left,
		up
	};

	for (fastUint episode{0}; episode < 100; ++episode) {
		std::cout << episode << " e\n";

		std::vector<fastUint> states{};
		std::vector<fastInt> rewards{};

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
					if (policyDistribution(generator) > .2f) {//.2: random action chance
						std::cout << "greedy\n";
						fastUint nextState;
						float maxValue;

						//set next state
						{
							//can move right
							if (canMoveRight(currentX)) {
								//right
								nextState = currentState + 1;
								setMaxValue(nextState, maxValue);

								//can move left
								if (currentX > 0) tryUpdateNextStateAndMaxValue(getLeftState(currentState), maxValue, nextState);

								trySetNextStateToDownOrUp(currentState, maxValue, nextState);
							} else {
								//left
								nextState = getLeftState(currentState);
								setMaxValue(nextState, maxValue);

								trySetNextStateToDownOrUp(currentState, maxValue, nextState);
							}
						}

						currentState = nextState;
					} else {//random action
						std::cout << "rand\n";

						std::vector<Action> possibleActions{};

						if (canMoveRight(currentX)) possibleActions.push_back(Action::right);

						//can move left
						if (currentX > 0) possibleActions.push_back(Action::left);

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
					}
				}

				++step;
			}
		}

		//values
		float Return{.0f};
		for (fastUint step{0}; step < states.size(); ++step) {
			Return = rewards[step] + .99f * Return;//.99: discount factor
			values[states[step]] += .1f * (Return - values[states[step]]);//.1: learning rate
		}
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