package com.ctliu3.hpc.neontest;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Message;

import com.ctliu3.hpc.neon.BenchTest;


public class RunningThread extends Thread {

  private Handler handler;

  public RunningThread(Handler handler) {
    this.handler = handler;
  }

  public void run() {
    long start = System.currentTimeMillis();
    String result = BenchTest.benchmark();
    long time = System.currentTimeMillis() - start;
    Message msg = handler.obtainMessage();
    msg.obj = result +
      "Java total time = " + time + "ms\n";
    handler.sendMessage(msg);
  }
}
