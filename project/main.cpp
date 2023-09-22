#include <cstdint>
#include <iostream>
#include <random>

using fastUint = std::uint_fast16_t;

fastUint currentState{0};
fastUint x{4};

fastUint currentX() { return currentState % x; }

int main() {
	fastUint step{0};
	fastUint y{4};
	bool goalAchieved{false};

	fastUint goal{x * y - 1};
	fastUint maxX{x - 1};

	std::mt19937 generator{std::random_device{}()};
	std::uniform_int_distribution actionsDistribution{0, 3};;

	while (!goalAchieved) {
		//current x = current position % x size
		//current y = current position // x size
		std::cout << step << ' ' << currentX() << ' ' << currentState / x << '\n';

		if (currentState == goal) {
			goalAchieved = true;
			continue;
		}

		enum class Action : fastUint {
			right,
			down,
			left,
			up
		};

		Action action{static_cast<Action>(actionsDistribution(generator))};

		switch (action) {
			case Action::right:
				if (currentX() < maxX) ++currentState;
				break;
			case Action::down: {
				fastUint nextState{currentState + x};
				if (nextState < goal) currentState = nextState;
				break;
			}
			case Action::left:
				if (currentX() % x > 0) --currentState;
				break;
			case Action::up: {
				using fastInt = std::int_fast16_t;
				fastInt nextState{static_cast<fastInt>(currentState) - static_cast<fastInt>(x)};
				if (nextState >= 0) currentState = nextState;
				break;
			}
		}

		++step;
	}
}