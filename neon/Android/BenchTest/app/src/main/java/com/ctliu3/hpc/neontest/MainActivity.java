package com.ctliu3.hpc.neontest;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


public class MainActivity extends AppCompatActivity {

  private Button btnBenchmark;
  private ImageView imageView;
  private TextView textView;
  private Handler handler = new Handler() {
    @Override
    public void handleMessage(Message msg) {
      if (msg.obj != null) {
        if (msg.obj instanceof Bitmap) {
          imageView.setImageBitmap((Bitmap) msg.obj);
        } else {
          btnBenchmark.setEnabled(true);
          textView.setText((String)msg.obj);
        }
      }
    }
  };
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    setContentView(R.layout.activity_main);
    btnBenchmark = (Button) findViewById(R.id.benchmark);
    imageView = (ImageView) findViewById(R.id.image);
    textView = (TextView) findViewById(R.id.info);

    btnBenchmark.setOnClickListener(view -> {
      btnBenchmark.setEnabled(false);
      new RunningThread(handler).start();
    });
  }

  @Override
  protected void onDestroy() {
    handler.removeCallbacksAndMessages(null);
    super.onDestroy();
  }
}
