#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using fastUint = std::uint_fast16_t;

constexpr fastUint x{4};

fastUint currentX(const fastUint currentState) { return currentState % x; }

int main() {
	constexpr fastUint y{5};

	constexpr fastUint maxX{x - 1};
	constexpr fastUint statesNum{x * y};
	constexpr fastUint goal{statesNum - 1};

	std::mt19937 generator{std::random_device{}()};
	std::uniform_int_distribution actionsDistribution{0, 3};

	enum class Action : fastUint {
		right,
		down,
		left,
		up
	};

	std::array<float, statesNum> values{};

	for (fastUint episode{0}; episode < 2; ++episode) {
		std::cout << episode << " e\n";

		std::vector<fastUint> states{};

		using fastInt = std::int_fast16_t;
		std::vector<fastInt> rewards{};

		//episode
		{
			fastUint currentState{0};
			fastUint step{0};
			while (currentState != goal) {
				//current x = current position % x size
				//current y = current position // x size
				std::cout << step << ' ' << currentX(currentState) << ' ' << currentState / x << '\n';

				states.push_back(currentState);
				rewards.push_back(-1);

				Action action{static_cast<Action>(actionsDistribution(generator))};

				switch (action) {
					case Action::right:
						if (currentX(currentState) < maxX) ++currentState;
						break;
					case Action::down: {
						fastUint nextState{currentState + x};
						if (nextState < goal) currentState = nextState;
						break;
					}
					case Action::left:
						if (currentX(currentState) % x > 0) --currentState;
						break;
					case Action::up: {
						fastInt nextState{static_cast<fastInt>(currentState) - static_cast<fastInt>(x)};
						if (nextState >= 0) currentState = nextState;
						break;
					}
				}

				++step;
			}
		}

		//values
		float Return{.0f};
		for (fastUint step{0}; step < states.size(); ++step) {
			Return = rewards[step] + .99f * Return;							  //.99: discount factor
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