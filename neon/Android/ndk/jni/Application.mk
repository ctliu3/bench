APP_STL := gnustl_static
# opencv need rtti and exceptions
APP_CPPFLAGS := -frtti -fexceptions
APP_ABI := armeabi-v7a
APP_PLATFORM := android-23
APP_CPPFLAGS += -std=c++11
APP_OPTIM := release
