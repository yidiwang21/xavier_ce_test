#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include "include.cuh"

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

void get_data_from_sensor(int src_file, std::string dest_file, int freq_div);

#endif