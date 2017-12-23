#include "jni.h"
#include <string>
#include "test.h"

extern "C" {


jstring benchmark_jni(JNIEnv *jenv, jclass thiz) {
  std::string ret = bench();
  return jenv->NewStringUTF(ret.c_str());
}

// for end of "extern C"
}

static const char *classPathName = "com/ctliu3/hpc/neon/BenchTest";


static JNINativeMethod methods[] =
{
    {"benchmark", "()Ljava/lang/String;", (void*)benchmark_jni },
};

/*
 * Register several native methods for one class.
 */
static int registerNativeMethods(JNIEnv* env, const char* className,
                                 JNINativeMethod* gMethods, int numMethods)
{
    jclass clazz;
    clazz = env->FindClass(className);
    if (clazz == NULL)
    {
        return JNI_FALSE;
    }
    if (env->RegisterNatives(clazz, gMethods, numMethods) < 0)
    {
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

/*
 * Register native methods for all classes we know about.
 *
 * returns JNI_TRUE on success.
 */
static int registerNatives(JNIEnv* env)
{
  if (!registerNativeMethods(env, classPathName, methods, sizeof(methods) / sizeof(methods[0])))
  {
    return JNI_FALSE;
  }

  return JNI_TRUE;
}


// ----------------------------------------------------------------------------

/*
 * This is called by the VM when the shared library is first loaded.
 */

#ifdef WIN32
JNIEXPORT jint JNICALL
#else
jint
#endif
JNI_OnLoad(JavaVM* vm, void* reserved)
{
	jint result = -1;
	JNIEnv* env = NULL;

	if (vm->GetEnv((void**)&env, JNI_VERSION_1_4) != JNI_OK)
	{
		goto bail;
	}
	if (registerNatives(env) != JNI_TRUE)
	{
		goto bail;
	}
	result = JNI_VERSION_1_4;
bail:
	return result;
}
