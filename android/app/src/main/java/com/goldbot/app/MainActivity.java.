package com.goldbot.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.text.InputType;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;

public class MainActivity extends Activity {
    private WebView web;
    private SharedPreferences prefs;
    private boolean dialogOpen = false;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        prefs = getSharedPreferences("goldbot", MODE_PRIVATE);
        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#0A0709"));
        setContentView(web);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView v, WebResourceRequest r, WebResourceError e) {
                if (r.isForMainFrame()) askUrl("Can't reach the server. Check the address:");
            }
        });
        String url = prefs.getString("url", "");
        if (url.isEmpty()) askUrl("Enter your Gold Bot server address:");
        else web.loadUrl(url);
    }

    private void askUrl(String msg) {
        if (dialogOpen) return;
        dialogOpen = true;
        final EditText in = new EditText(this);
        in.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        in.setHint("https://your-server.example.com");
        in.setText(prefs.getString("url", ""));
        new AlertDialog.Builder(this)
            .setTitle("Gold Bot server").setMessage(msg).setView(in).setCancelable(false)
            .setPositiveButton("Connect", (d, w) -> {
                String u = in.getText().toString().trim();
                if (!u.startsWith("http")) u = "https://" + u;
                prefs.edit().putString("url", u).apply();
                dialogOpen = false;
                web.loadUrl(u);
            })
            .setNegativeButton("Exit", (d, w) -> finish())
            .show();
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack(); else super.onBackPressed();
    }
}
