#ifndef PP_TIMER_H_
#define PP_TIMER_H_

#include <ctime>

namespace pp {

typedef struct timespec Time;

Time GetZeroTime();
Time GetCurrentTime();

Time TimeAdd(Time base, Time add);
Time TimeDiff(Time start, Time end);
long TimeDiffInMs(Time start, Time end);

long TimeToLongInMs(Time t);

} // namespace pp

#endif
