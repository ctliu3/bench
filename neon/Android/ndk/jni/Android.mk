LOCAL_PATH := $(call my-dir)
MY_PATH := $(LOCAL_PATH)

SRC_FILE_PATH := $(LOCAL_PATH)/../../../bench

include $(CLEAR_VARS)

LOCAL_SRC_FILES  := \
$(SRC_FILE_PATH)/src/hidden_mem_lantency.cpp \
main.cpp \
test.cpp

LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_C_INCLUDES += $(SRC_FILE_PATH)
LOCAL_C_INCLUDES += $(SRC_FILE_PATH)/include

LOCAL_LDLIBS    += -llog -ldl -ljnigraphics
LOCAL_CFLAGS    += -mfpu=neon -std=c++11
LOCAL_CFLAGS    += -fexceptions
LOCAL_CFLAGS    += -ffast-math -Os -funroll-loops
LOCAL_ARM_NEON  := true
LOCAL_ARM_MODE  := arm
LOCAL_MODULE    := benchtest

include $(BUILD_SHARED_LIBRARY)
