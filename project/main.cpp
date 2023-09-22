#include <iostream>
#include <cstdint>

std::uint_fast16_t currentState{0};
std::uint_fast16_t x{4};

std::uint_fast16_t currentX() { return currentState % x; }

int main() {
	std::uint_fast16_t step{0};
	std::uint_fast16_t y{4};
	bool goalAchieved{false};

	std::uint_fast16_t goal{static_cast<std::uint_fast16_t>(x * y - 1)};
	std::uint_fast16_t maxX{static_cast<std::uint_fast16_t>(x - 1)};

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
				std::uint_fast16_t nextState{static_cast<std::uint_fast16_t>(currentState + x)};
				if (nextState < goal) currentState = nextState;
				break;
			}
			case action::left:
				if (currentX() % x > 0) --currentState;
				break;
			case action::up: {
				std::int_fast16_t nextState{static_cast<std::int_fast16_t>(currentState) - static_cast<std::int_fast16_t>(x)};
				if (nextState >= 0) currentState = nextState;
				break;
			}
		}

		++step;
	}
}