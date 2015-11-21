#ifndef PP_TIMER_H_
#define PP_TIMER_H_

#include <ctime>

namespace pp {

typedef struct timespec Time;

Time GetCurrentTime();

Time TimeDiff(Time start, Time end);
long TimeDiffInMs(Time start, Time end);

} // namespace pp

#endif
