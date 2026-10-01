import base64, glob, os, time

html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>DFL Live Preview</title>
  <style>
    body {{ background: #121212; color: #eee; font-family: sans-serif; text-align: center; margin: 10px; }}
    .card {{ margin: 16px auto; max-width: 95%; background: #1e1e1e; border-radius: 8px; padding: 12px; }}
    img {{ max-width: 100%; border-radius: 4px; display: block; margin: 0 auto; }}
    .title {{ font-weight: bold; margin-bottom: 8px; font-size: 14px; color: #aaa; }}
    .header-box {{ display: flex; justify-content: center; align-items: center; gap: 14px; margin-bottom: 12px; }}
    .timer-badge {{ background: #2a2a2a; border: 1px solid #444; border-radius: 20px; padding: 6px 14px; font-size: 13px; font-weight: bold; color: #4af626; }}
    .status {{ font-size: 12px; color: #888; }}
  </style>
</head>
<body>
  <h2>DeepFaceLab Live Preview</h2>
  <div class="header-box">
    <div class="timer-badge">Next refresh: <span id="countdown">30</span>s</div>
    <div class="status">Last generated: {time_str}</div>
  </div>
  {cards}

  <script>
    let secondsLeft = 30;
    const el = document.getElementById("countdown");
    setInterval(() => {{
      secondsLeft--;
      if (secondsLeft <= 0) {{
        el.innerText = "0";
        location.reload();
      }} else {{
        el.innerText = secondsLeft;
      }}
    }}, 1000);
  </script>
</body>
</html>"""

while True:
    cards = ""
    for path in sorted(glob.glob("/workspace/model/*preview*.jpg")):
        name = os.path.basename(path).replace(".jpg", "")
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            cards += f"<div class=\"card\"><div class=\"title\">{name}</div><img src=\"data:image/jpeg;base64,{b64}\"></div>"
        except Exception:
            pass

    full_html = html_template.format(time_str=time.strftime("%T"), cards=cards)
    with open("/workspace/live_preview.html", "w") as f:
        f.write(full_html)
    time.sleep(30)
