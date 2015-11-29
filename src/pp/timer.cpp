#include "timer.hpp"

namespace pp {

Time GetZeroTime() {
	Time t;
	t.tv_sec = 0;
	t.tv_nsec = 0;
	return t;
}

Time GetCurrentTime() {
	Time t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t;
}

Time TimeAdd(Time base, Time add) {
	base.tv_sec += add.tv_sec;
	base.tv_nsec += add.tv_nsec;
	if (base.tv_nsec >= 1000000000) {
		base.tv_sec++;
		base.tv_nsec -= 1000000000;
	}
	return base;
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
	return TimeToLongInMs(diff);
}

long TimeToLongInMs(Time t) {
	long ms;

	// Convert it to ms.
	ms = t.tv_nsec / 1000000;
	ms += t.tv_sec * 1000;

	return ms;
}

} // namespace pp
