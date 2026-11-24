"""Local web demo (stdlib only). Uses predict.py without labels; opens in the default browser."""
import argparse
import base64
import json
import tempfile
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from predict import predict
from object_counting.io import ROOT, load_scene

MAX_UPLOAD = 50 * 1024 * 1024


def run_prediction(scene, config, name='', data=b'', path=''):
    """Predict an uploaded image (data) or a local file (path). Returns a JSON-ready dict."""
    if data:
        with tempfile.TemporaryDirectory() as tmp:
            # Keep the original file name: predict's test-freeze guard also checks the sha256, so renaming cannot bypass it.
            image = Path(tmp) / (Path(name).name or 'upload.png')
            image.write_bytes(data)
            result, out, scene_id = predict(str(image), scene, config)
    elif path.strip():
        result, out, scene_id = predict(path.strip(), scene, config)
    else:
        raise ValueError('Please choose an image before counting.')
    image_b64 = lambda f: base64.b64encode((out / f).read_bytes()).decode()
    return dict(count=result.count, scene_id=scene_id, ms=round(result.timings_ms['pipeline'], 1),
                background_status=load_scene(scene_id)[0].get('background_status', 'unknown'),
                output=str(out), overlay=image_b64('overlay.png'), panel=image_b64('panel.png'))


class Handler(BaseHTTPRequestHandler):
    defaults = {}

    def send(self, code, body, kind):
        self.send_response(code); self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path != '/':
            return self.send(404, b'Not found', 'text/plain')
        configs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'configs').glob('*.toml'))
        scenes = ['auto'] + sorted(p.stem for p in (ROOT / 'configs/scenes').glob('*.toml'))
        page = PAGE.replace('__DATA__', json.dumps(dict(configs=configs, scenes=scenes, **self.defaults)))
        self.send(200, page.encode(), 'text/html; charset=utf-8')

    def do_POST(self):
        if urlparse(self.path).path != '/predict':
            return self.send(404, b'Not found', 'text/plain')
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        try:
            size = int(self.headers.get('Content-Length') or 0)
            if size > MAX_UPLOAD:
                raise ValueError('Image too large (max 50 MB).')
            data = self.rfile.read(size) if size else b''
            body = run_prediction(q.get('scene', 'auto'), q.get('config', 'configs/final.toml'),
                                  q.get('name', ''), data, q.get('path', ''))
            self.send(200, json.dumps(body).encode(), 'application/json')
        except Exception as error:
            self.send(400, json.dumps(dict(error=str(error))).encode(), 'application/json')

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default=''); parser.add_argument('--scene', default='auto')
    parser.add_argument('--config', default='configs/final.toml')
    parser.add_argument('--port', type=int, default=8765); parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    Handler.defaults = dict(image=args.image, scene=args.scene, config=args.config)
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}/'
    print('Demo running at', url, '— Ctrl+C to stop.')
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Object Counting · Image Lab</title>
<style>
:root{--bg:#eef1f6;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#2563eb;--accent2:#1d4ed8;--ok:#15803d;--err:#b91c1c}
*{box-sizing:border-box}
body{margin:0;font-family:"Inter","Segoe UI","Noto Sans","DejaVu Sans",system-ui,sans-serif;background:var(--bg);color:var(--ink);font-size:14px}
header{background:linear-gradient(120deg,#0f172a,#1e3a8a);color:#fff;padding:20px 28px}
header h1{margin:0;font-size:22px;font-weight:700}
header p{margin:4px 0 0;color:#cbd5e1;font-size:13px}
main{display:grid;grid-template-columns:340px 1fr;gap:18px;padding:18px 28px;max-width:1500px;margin:auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 1px 3px rgba(15,23,42,.06)}
.card+.card{margin-top:18px}
h2{font-size:15px;margin:0 0 14px}
label{display:block;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:12px 0 5px}
select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:9px;font:inherit;background:#fff}
select:focus{outline:2px solid #bfdbfe;border-color:var(--accent)}
#drop{border:2px dashed #cbd5e1;border-radius:12px;padding:16px;text-align:center;cursor:pointer;color:var(--muted);transition:.15s}
#drop:hover,#drop.over{border-color:var(--accent);background:#eff6ff;color:var(--accent)}
#drop img{max-width:100%;max-height:150px;border-radius:8px;display:none;margin:0 auto 8px}
#drop .name{font-weight:600;color:var(--ink);word-break:break-all}
button.run{width:100%;margin-top:16px;padding:12px;border:0;border-radius:10px;background:var(--accent);color:#fff;font-family:inherit;font-size:15px;font-weight:600;cursor:pointer}
button.run:hover{background:var(--accent2)}button.run:disabled{background:#93c5fd;cursor:wait}
.hint{text-align:center;color:var(--muted);font-size:12px;margin-top:8px}
.count{display:flex;align-items:baseline;gap:10px}
.count b{font-size:56px;line-height:1;color:var(--accent)}.count span{color:var(--muted);font-size:15px}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;margin:14px 0}dt{color:var(--muted)}dd{margin:0;font-weight:600}
.msg{font-size:13px;padding:10px 12px;border-radius:9px;background:#f8fafc;color:var(--muted)}
.msg.ok{background:#f0fdf4;color:var(--ok)}.msg.err{background:#fef2f2;color:var(--err)}
.view{display:flex;flex-direction:column;min-height:560px}
.viewhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.viewhead h2{margin:0}
.tabs button{border:1px solid var(--line);background:#fff;padding:6px 12px;font:inherit;cursor:pointer;color:var(--muted)}
.tabs button:first-child{border-radius:8px 0 0 8px}.tabs button:last-child{border-radius:0 8px 8px 0;border-left:0}
.tabs button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.stage{flex:1;display:flex;align-items:center;justify-content:center;background:#f8fafc;border-radius:10px;overflow:hidden}
.stage img{width:100%;height:75vh;object-fit:contain;display:none}
.empty{color:var(--muted);text-align:center;line-height:1.7}
footer{padding:0 28px 18px;color:var(--muted);font-size:12px;max-width:1500px;margin:auto;word-break:break-all}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #fff;border-right-color:transparent;border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<header><h1>People Counting from Surveillance Images</h1><p>New images need no labels · Requires a background image from the same camera angle</p></header>
<main>
<section>
 <div class="card"><h2>Settings</h2>
  <label for="scene">Camera scene</label><select id="scene"></select>
  <label>Input image</label>
  <div id="drop" tabindex="0"><img id="thumb" alt=""><div class="name" id="fname"></div><div id="dhint">Drag and drop an image here or click to choose</div></div>
  <input type="file" id="file" accept="image/*" hidden>
  <label for="config">Configuration</label><select id="config"></select>
  <button class="run" id="run">Count</button>
  <div class="hint">Ctrl+O choose image · Ctrl+Enter run</div>
 </div>
 <div class="card"><h2>Result</h2>
  <div class="count"><b id="count">—</b><span>people (predicted)</span></div>
  <dl><dt>Scene</dt><dd id="r-scene">—</dd><dt>Processing time</dt><dd id="r-ms">—</dd><dt>Background status</dt><dd id="r-bg">—</dd></dl>
  <div class="msg" id="msg">Choose the matching scene and image. Test images (scene_03, holdout vtest) only run after the freeze-test tag.</div>
 </div>
</section>
<section class="card view">
 <div class="viewhead"><h2>Output image</h2><div class="tabs"><button class="on" data-k="overlay">Result</button><button data-k="panel">Processing steps</button></div></div>
 <div class="stage"><img id="out" alt="Output image"><div class="empty" id="empty">No output yet<br>Choose an image, then click “Count”</div></div>
</section>
</main>
<footer id="saved">Ready</footer>
<script>
const D=__DATA__, $=id=>document.getElementById(id);
let file=null, path=D.image||'', last=null, tab='overlay';
const fill=(sel,vals,cur)=>{for(const v of vals){const o=new Option(v,v);o.selected=v===cur;sel.add(o)}if(cur&&!vals.includes(cur))sel.add(new Option(cur,cur,true,true))};
fill($('scene'),D.scenes,D.scene);fill($('config'),D.configs,D.config);
if(path){$('fname').textContent=path;$('dhint').textContent='Click to change image'}
function pick(f){if(!f)return;file=f;path='';$('fname').textContent=f.name;$('dhint').textContent='Click to change image';
 const t=$('thumb');t.src=URL.createObjectURL(f);t.style.display='block'}
$('drop').onclick=()=>$('file').click();
$('drop').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click()}};
$('file').onchange=e=>pick(e.target.files[0]);
['dragenter','dragover'].forEach(n=>$('drop').addEventListener(n,e=>{e.preventDefault();$('drop').classList.add('over')}));
['dragleave','drop'].forEach(n=>$('drop').addEventListener(n,e=>{e.preventDefault();$('drop').classList.remove('over')}));
$('drop').addEventListener('drop',e=>pick(e.dataTransfer.files[0]));
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{tab=b.dataset.k;document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));show()});
function show(){const img=$('out');if(!last){img.style.display='none';$('empty').style.display='block';return}
 img.src='data:image/png;base64,'+last[tab];img.style.display='block';$('empty').style.display='none'}
function msg(t,c){$('msg').textContent=t;$('msg').className='msg '+(c||'')}
async function run(){const b=$('run');b.disabled=true;b.innerHTML='<span class="spin"></span>Processing…';msg('Processing…');
 const q=new URLSearchParams({scene:$('scene').value,config:$('config').value});
 if(file)q.set('name',file.name);else q.set('path',path);
 try{const r=await fetch('/predict?'+q,{method:'POST',body:file||''}),j=await r.json();if(!r.ok)throw new Error(j.error);
  last=j;$('count').textContent=j.count;$('r-scene').textContent=j.scene_id;$('r-ms').textContent=j.ms+' ms';$('r-bg').textContent=j.background_status;
  msg('✓ Done — the count is a prediction, not a ground-truth label.','ok');$('saved').textContent='Result saved to: '+j.output}
 catch(e){last=null;$('count').textContent='—';['r-scene','r-ms','r-bg'].forEach(i=>$(i).textContent='—');
  msg('✗ Prediction failed: '+e.message,'err');$('saved').textContent='Ready'}
 finally{b.disabled=false;b.textContent='Count';show()}}
$('run').onclick=run;
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')run();if(e.ctrlKey&&e.key.toLowerCase()==='o'){e.preventDefault();$('file').click()}});
</script></body></html>'''


if __name__ == '__main__':
    main()
