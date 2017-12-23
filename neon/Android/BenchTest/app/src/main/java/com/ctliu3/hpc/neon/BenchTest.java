package com.ctliu3.hpc.neon;

public class BenchTest {
    static {
        System.loadLibrary("benchtest");
    }
    public static native String benchmark();
}