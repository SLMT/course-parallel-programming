#include "timer.hpp"

namespace pp {

Time GetCurrentTime() {
	Time t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t;
}

Time TimeDiff(Time start, Time end) {
	Time tmp;

	// Calculate diff
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        tmp.tv_sec = end.tv_sec - start.tv_sec - 1;
        tmp.tv_nsec = end.tv_nsec - start.tv_nsec + 1000000000;
    } else {
        tmp.tv_sec = end.tv_sec - start.tv_sec;
        tmp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }

	return tmp;
}

long TimeDiffInMs(Time start, Time end) {
	Time diff = TimeDiff(start, end);
	long ms;

	// Convert it to ms.
	ms = diff.tv_nsec / 1000000;
	ms += diff.tv_sec * 1000;

	return ms;
}

} // namespace pp
