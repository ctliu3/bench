#include "bench.h"
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <arm_neon.h>

void AddNavie(int8_t* a, int8_t* b, int16_t* c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

void AddNeonInt8(int8_t* a, int8_t* b, int16_t* c, int n) {
  int i = 0;
  for (; i < n; i += 8) {
    int8x8_t va = vld1_s8(a + i);
    int8x8_t vb = vld1_s8(b + i);
    int16x8_t vc = vaddl_s8(va, vb);
    vst1q_s16(c + i, vc);
  }
  for (; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

void AddNeonInt8_V2(int8_t* a, int8_t* b, int16_t* c, int n) {
  int i = 0;
  for (; i < n; i += 32) {
    int8x8x4_t v, v2;
    v.val[0] = vld1_s8(a + i);
    v.val[1] = vld1_s8(b + i);
    v.val[2] = vld1_s8(a + i + 8);
    v.val[3] = vld1_s8(b + i + 8);
    v2.val[0] = vld1_s8(a + i + 16);
    v2.val[1] = vld1_s8(b + i + 16);
    v2.val[2] = vld1_s8(a + i + 24);
    v2.val[3] = vld1_s8(b + i + 24);
    vst1q_s16(c + i, vaddl_s8(v.val[0], v.val[1]));
    vst1q_s16(c + i + 8, vaddl_s8(v.val[2], v.val[3]));
    vst1q_s16(c + i + 16, vaddl_s8(v2.val[0], v2.val[1]));
    vst1q_s16(c + i + 24, vaddl_s8(v2.val[2], v2.val[3]));
  }
  for (; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

void AddNeonInt8_V3(int8_t* a, int8_t* b, int16_t* c, int n) {
  int i = 0;
  for (; i < n; i += 32) {
    int8x8x4_t v, v2;
    v.val[0] = vld1_s8(a + i);
    v.val[1] = vld1_s8(b + i);
    vst1q_s16(c + i, vaddl_s8(v.val[0], v.val[1]));
    v.val[2] = vld1_s8(a + i + 8);
    v.val[3] = vld1_s8(b + i + 8);
    vst1q_s16(c + i + 8, vaddl_s8(v.val[2], v.val[3]));
    v2.val[0] = vld1_s8(a + i + 16);
    v2.val[1] = vld1_s8(b + i + 16);
    vst1q_s16(c + i + 16, vaddl_s8(v2.val[0], v2.val[1]));
    v2.val[2] = vld1_s8(a + i + 24);
    v2.val[3] = vld1_s8(b + i + 24);
    vst1q_s16(c + i + 24, vaddl_s8(v2.val[2], v2.val[3]));
  }
  for (; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

int GetDiff(int8_t* a, int8_t* b, int16_t* c, int size) {
  int count = 0;
  for (int i = 0; i < size; ++i) {
    if (int(c[i]) != (int)a[i] + b[i]) {
      ++count;
    }
  }
  return count;
}

void arr_add() {
  int times = 1000;
  int size = 1024 * 1024;
  int8_t * a = (int8_t*)malloc(sizeof(int8_t) * size);
  int8_t * b = (int8_t*)malloc(sizeof(int8_t) * size);

  srand(time(NULL));
  for (int i = 0; i < size; ++i) {
    a[i] = rand() % INT8_MAX;
    b[i] = rand() % INT8_MAX;
  }

  int16_t * c = (int16_t*)malloc(sizeof(int16_t) * size);
  memset(c, 0, sizeof(int16_t) * size);

  int st, ed, count;
  st = GetCurrentMicrosecond();
  for (int i = 0; i < times; ++i) {
    // AddNeonInt8(a, b, c, size);
    AddNavie(a, b, c, size);
//     AddNeonInt8_V2(a, b, c, size);
  }
  ed = GetCurrentMicrosecond();
  int elapsed1 = ed - st;

  count = GetDiff(a, b, c, size);
  LOGD("=== int8: %dms, diff = %d", elapsed1, count);

  memset(c, 0, sizeof(int16_t) * size);
  st = GetCurrentMicrosecond();
  for (int i = 0; i < times; ++i) {
//     AddNeonInt8(a, b, c, size);
//    AddNeonInt8_V2(a, b, c, size);
    AddNeonInt8_V3(a, b, c, size);
  }
  ed = GetCurrentMicrosecond();
  int elapsed2 = ed - st;

  count = GetDiff(a, b, c, size);
  LOGD("=== int8 v3: %dms, ratio: %.2f, diff = %d", elapsed2, (elapsed1 - elapsed2) * 1. /elapsed1, count);
}
