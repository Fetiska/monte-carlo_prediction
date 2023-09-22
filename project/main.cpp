#include <iostream>
#include <cstdint>

using fastUint = std::uint_fast16_t;

fastUint currentState{0};
fastUint x{4};

fastUint currentX() { return currentState % x; }

int main() {
	fastUint step{0};
	fastUint y{4};
	bool goalAchieved{false};

	fastUint goal{static_cast<fastUint>(x * y - 1)};
	fastUint maxX{static_cast<fastUint>(x - 1)};

	while (!goalAchieved) {
		//current x = current position % x size
		//current y = current position // x size
		std::cout << step << ' ' << currentX() << ' ' << currentState / x << '\n';

		if (currentState == goal) {
			goalAchieved = true;
			continue;
		}

		enum class action {
			right,
			down,
			left,
			up
		};

		action action{action::up};

		switch (action) {
			case action::right:
				if (currentX() < maxX) ++currentState;
				break;
			case action::down: {
				fastUint nextState{static_cast<fastUint>(currentState + x)};
				if (nextState < goal) currentState = nextState;
				break;
			}
			case action::left:
				if (currentX() % x > 0) --currentState;
				break;
			case action::up: {
				using fastInt = std::int_fast16_t;

				fastInt nextState{static_cast<fastInt>(currentState) - static_cast<fastInt>(x)};
				if (nextState >= 0) currentState = nextState;
				break;
			}
		}

		++step;
	}
}