#include "timedaction.h"
#include <stdio.h>

struct timespec start;
#define MS (1000000)

void hey_cb(void* data) {
    fprintf(stderr, "Hey %d\n", *(int *) data);
}

int main(void) {
   timed_action_notifier* notifier = timed_action_mainloop_threaded();

    int dat = 2;
    timed_action_schedule_periodic(notifier, 1, 0, &hey_cb, &dat);

    sleep(20);
    return 0;
}
