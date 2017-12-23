#ifndef _UTILITY_H
#define _UTILITY_H

#include <time.h>

#ifdef __ANDROID__

#include <android/log.h>

#define DEBUG_LOG 1

#ifndef LOG_TAG
#  define LOG_TAG "Bench"
#endif

#ifdef DEBUG_LOG
#  define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#  define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#  define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#else
#  define LOGD(...)
#  define LOGI(...)
#  define LOGE(...)
#endif

#else // __ANDROID__
#  define LOGD(...)
#  define LOGI(...)
#  define LOGE(...)
#endif


/* return current time in milliseconds */

inline int GetCurrentMillisecond() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<int>(static_cast<uint64_t>(t.tv_sec) * 1000 +
         static_cast<uint64_t>(t.tv_nsec) / 1000000);
}


/* return current time in microseconds */
inline int GetCurrentMicrosecond() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<int>(static_cast<uint64_t>(t.tv_sec) * 1000000 +
         static_cast<uint64_t>(t.tv_nsec) / 1000);
}


#endif // _UTILITY_H
