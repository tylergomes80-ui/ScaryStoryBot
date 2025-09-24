#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Super Background Downloader Pro++ — God-Mode Edition (hardened)
- Fixes ffmpeg quoting: drawtext via textfile, safe path escaping, filter_complex for watermark.
- Auto-detects yt-dlp/ffmpeg/ffprobe from local, env, or PATH.
- Caches searches, filters on duration, enforces quotas (50 GB).
- Retries with backoff, CUDA accel if available.
- Randomized hook overlays + optional watermark.
- Loudness normalize + silence trim.
- Ambience / gameplay categories, CLI args, interactive menu.
- Metadata export and robust logging.
"""

import os, sys, json, random, subprocess, shutil, time, yaml, argparse, logging, tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Paths
# =========================
BASE_DIR    = Path(__file__).parent
BG_DIR      = BASE_DIR / "backgrounds"
CACHE_DIR   = BASE_DIR / "cache_videos"
TEMP_DIR    = BASE_DIR / "temp_downloads"
LOG_DIR     = BASE_DIR / "logs"
ASSETS_DIR  = BASE_DIR / "assets"
CONFIG_PATH = BASE_DIR / "config.yaml"

for d in [BG_DIR, CACHE_DIR, TEMP_DIR, LOG_DIR, ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Tool detection
# =========================
def sanitize_path(p): return str(p).strip().strip('"').strip("'")

local_yt = BASE_DIR / "yt-dlp.exe"
if local_yt.exists():
    YTDLP = str(local_yt)
elif os.environ.get("YTDLP"):
    YTDLP = sanitize_path(os.environ["YTDLP"])
else:
    YTDLP = "yt-dlp"

FFMPEG  = sanitize_path(os.environ.get("FFMPEG", "ffmpeg"))
FFPROBE = sanitize_path(os.environ.get("FFPROBE", "ffprobe"))
WATERMARK = str(ASSETS_DIR / "watermark.png")
MAX_BG_GB = 50

# =========================
# Logging
# =========================
log_path = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_path, encoding="utf-8")]
)
log = logging.getLogger("bgdl")

def check_tools():
    for tool, name in [(YTDLP, "yt-dlp"), (FFMPEG, "ffmpeg"), (FFPROBE, "ffprobe")]:
        if shutil.which(tool) or Path(tool).exists():
            log.info(f"{name} found: {tool}")
        else:
            log.warning(f"{name} NOT found: {tool}")
check_tools()

# =========================
# Config
# =========================
DEFAULT_CONFIG = {
    "categories": {
        "minecraft":   ["minecraft parkour gameplay no commentary"],
        "subway":      ["subway surfers gameplay no commentary"],
        "satisfying":  ["asmr slime cutting no talking"],
        "food":        ["cooking cutting food preparation tiktok background"]
    },
    "duration": {
        "minecraft": [30, 600],
        "subway":    [30, 600],
        "satisfying":[15, 300],
        "food":      [15, 600]
    },
    "cache_size": 200,
    "cache_days": 7,
    "title_blacklist": ["intro","outro","subscribe","compilation","credits"],
    "title_whitelist": ["satisfying","asmr","viral","epic"]
}

def ensure_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(yaml.dump(DEFAULT_CONFIG), encoding="utf-8")
        log.info("config.yaml not found. Created default config.")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    # minimal key guard
    for k,v in DEFAULT_CONFIG.items():
        if k not in cfg: cfg[k]=v
    return cfg

CONFIG = ensure_config()

# =========================
# Index
# =========================
INDEX_PATH = CACHE_DIR / "videos_index.json"
META_PATH  = CACHE_DIR / "export_metadata.json"

def load_index():
    try:
        if INDEX_PATH.exists():
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Index load failed: {e}")
    return {"downloaded": {}}

def save_index(ix): INDEX_PATH.write_text(json.dumps(ix, indent=2), encoding="utf-8")

# =========================
# Utils
# =========================
def run_cmd(cmd, retries=3, timeout=300, backoff=2.0, capture=False):
    for i in range(retries):
        try:
            if capture:
                res = subprocess.run(cmd, check=True, timeout=timeout,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return True, (res.stdout or ""), (res.stderr or "")
            else:
                subprocess.run(cmd, check=True, timeout=timeout)
                return True, "", ""
        except Exception as e:
            sleep_s = backoff ** (i+1)
            log.warning(f"cmd fail {e}. retry {i+1}/{retries} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return False, "", ""

def has_cuda():
    ok, out, _ = run_cmd([FFMPEG,"-hwaccels"], retries=1, timeout=10, capture=True)
    return ok and "cuda" in out.lower()
USE_CUDA = has_cuda()

def cleanup_temp():
    for p in TEMP_DIR.glob("*"):
        if p.is_file(): p.unlink(missing_ok=True)
        elif p.is_dir(): shutil.rmtree(p, ignore_errors=True)
    log.info("Temp cleaned.")

def enforce_quota(path, limit_gb=MAX_BG_GB):
    files = sorted(path.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files) / (1024**3)
    while total > limit_gb and files:
        f = files.pop(0)
        total -= f.stat().st_size / (1024**3)
        try:
            f.unlink(missing_ok=True)
            log.warning(f"Deleted {f.name} to enforce quota")
        except Exception as e:
            log.warning(f"Delete failed: {e}")
            break
    if total > limit_gb: log.warning(f"Still over quota {total:.2f} GB")

def get_duration(filepath):
    ok, out, _ = run_cmd([FFPROBE,"-v","error","-show_entries","format=duration",
                          "-of","default=noprint_wrappers=1:nokey=1", str(filepath)],
                          retries=1, timeout=20, capture=True)
    try: return float(out.strip())
    except: return 0.0

# =========================
# FFmpeg filter escaping helpers
# =========================
def ff_escape_path_for_filter(p: Path) -> str:
    """
    Convert a filesystem path to an ffmpeg filter-safe token:
    - use forward slashes
    - escape ':' as '\:'
    """
    s = str(p)
    s = s.replace("\\", "/")
    s = s.replace(":", r"\:")
    return s

def make_hook_textfile(text: str) -> Path:
    """
    Write the hook text to a temp file so drawtext never breaks on quotes/apostrophes.
    """
    # moviepy temp dir under TEMP_DIR
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="hook_", suffix=".txt", dir=str(TEMP_DIR))
    os.close(fd)
    Path(tmp).write_text(text, encoding="utf-8")
    return Path(tmp)

# =========================
# Hooks
# =========================
HOOKS = [
    ("DON'T SCROLL...",       "red",    "top"),
    ("WAIT UNTIL THE END",    "yellow", "top"),
    ("THIS GETS CREEPY",      "white",  "top"),
    ("TURN UP THE VOLUME",    "red",    "top"),
    ("ARE YOU ALONE?",        "yellow", "top"),
    ("KEEP WATCHING",         "white",  "top"),
]

def hook_drawtext_filter(textfile_path: Path, color: str, pos: str, font_size: int = 80) -> str:
    y = {"mid": "(h-text_h)/2", "bottom": "h-text_h-80"}.get(pos, "50")
    tf = ff_escape_path_for_filter(textfile_path)
    # enable uses comma which must be escaped as '\,' inside filter string
    enable = "lt(t\\,1)"
    return f"drawtext=textfile='{tf}':fontcolor={color}:fontsize={font_size}:x=(w-text_w)/2:y={y}:enable='{enable}'"

def watermark_chain(label_in: str, wm_input_label: str, w: int = 20, h: int = 20) -> str:
    """
    Build overlay step. Returns e.g. "[{label_in}][{wm_input_label}]overlay=W-w-20:H-h-20:enable='between(t,0,5)'[vout]"
    """
    return f"{label_in}{wm_input_label}overlay=W-w-{w}:H-h-{h}:enable='between(t,0,5)'[vout]"

# =========================
# Slice (robust filter_complex, no broken quoting)
# =========================
def slice_clip(filepath, outdir):
    dur = get_duration(filepath)
    base_vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"

    if dur < 30:
        loop_file = outdir / f"{Path(filepath).stem}_loop.mp4"
        cmd = [FFMPEG]
        if USE_CUDA: cmd += ["-hwaccel","cuda"]
        cmd += ["-stream_loop","-1","-i",str(filepath),
                "-t","30","-vf",base_vf,"-af","loudnorm",
                "-c:v","h264_nvenc" if USE_CUDA else "libx264",
                "-c:a","aac","-ar","44100","-y",str(loop_file)]
        run_cmd(cmd, retries=2, timeout=300)
        Path(filepath).unlink(missing_ok=True)
        return {"variant":"loop30"}

    seg_len = random.choice([30, 45, 60, 75]) if dur > 90 else int(dur)
    hook_text, color, pos = random.choice(HOOKS)
    font_size = random.choice([72,80,88])

    # Prepare hook textfile for drawtext
    hook_tf = make_hook_textfile(hook_text)
    hook_vf = hook_drawtext_filter(hook_tf, color, pos, font_size)

    # Build filter graph
    wm_exists = Path(WATERMARK).exists()
    filter_graph = ""

    if wm_exists:
        # Inputs: [0:v]=main, [1:v]=watermark (still)
        wm_fade = "format=rgba,fade=t=in:st=0:d=0.5,fade=t=out:st=4.5:d=0.5"
        filter_graph = (
            f"[0:v]{base_vf},{hook_vf}[v0];"
            f"[1:v]{wm_fade}[wm];"
            f"[v0][wm]overlay=W-w-20:H-h-20:enable='between(t,0,5)'[vout]"
        )
    else:
        filter_graph = f"[0:v]{base_vf},{hook_vf}[vout]"

    af_chain = "loudnorm,silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.5"
    outtmpl = str(outdir / f"{Path(filepath).stem}_part%03d.mp4")

    cmd = [FFMPEG]
    if USE_CUDA: cmd += ["-hwaccel","cuda"]
    cmd += ["-i", str(filepath)]
    if wm_exists:
        cmd += ["-loop","1","-i", WATERMARK]  # still image as second input
    cmd += [
        "-filter_complex", filter_graph,
        "-map","[vout]","-map","0:a?","-af", af_chain,
        "-segment_time", str(seg_len), "-f", "segment",
        "-c:v", "h264_nvenc" if USE_CUDA else "libx264",
        "-c:a", "aac", "-ar", "44100", outtmpl
    ]

    run_cmd(cmd, retries=2, timeout=900)
    # cleanup temp hook file
    with contextlib.suppress(Exception):
        Path(hook_tf).unlink(missing_ok=True)
    Path(filepath).unlink(missing_ok=True)
    return {"variant":"slice","seg_len":seg_len,"hook":hook_text}

# =========================
# Pipeline
# =========================
def process_video(url, cat, ix, meta):
    ok, _, serr = run_cmd(
        [YTDLP,"-f","bestvideo[height<=1080][fps<=60]+bestaudio/best",
         "-o",str(TEMP_DIR / f"{cat}_%(id)s.%(ext)s"),
         "--merge-output-format","mp4",url],
        retries=3,timeout=600,capture=True
    )
    if not ok:
        log.error(f"yt-dlp failed: {url} :: {serr[:200]}")
        return
    files = list(TEMP_DIR.glob(f"{cat}_*.mp4"))
    if not files:
        log.warning(f"No files downloaded for {url}")
        return
    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(slice_clip,f,BG_DIR) for f in files]
        for fut in as_completed(futs):
            try: results.append(fut.result())
            except Exception as e: log.warning(f"slice error: {e}")
    ix["downloaded"][url] = {"cat":cat,"url":url}
    save_index(ix)
    meta.append({"src":url,"cat":cat,"created":datetime.now().isoformat(),"results":results})

def download(cat, count, ix, meta):
    picks = []
    for q in CONFIG["categories"][cat]:
        ok, sout, _ = run_cmd([YTDLP,f"ytsearch{CONFIG['cache_size']}:{q}",
                               "--skip-download","--print-json","--dateafter","20240101"],
                               retries=2,timeout=600,capture=True)
        if ok:
            for line in sout.splitlines():
                try:
                    j=json.loads(line.strip())
                    if j.get("_type")=="playlist": continue
                    dur=j.get("duration") or 0
                    if CONFIG["duration"][cat][0]<=dur<=CONFIG["duration"][cat][1]:
                        picks.append(j)
                except: continue
    random.shuffle(picks)
    picks=picks[:count]
    if not picks:
        log.info(f"No candidates for {cat}")
        return
    urls=[p["webpage_url"] for p in picks if p.get("webpage_url")]
    log.info(f"{cat}: downloading {len(urls)}")
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs=[ex.submit(process_video,u,cat,ix,meta) for u in urls]
        for _ in as_completed(futs): pass
    enforce_quota(BG_DIR)

# =========================
# Menu
# =========================
def menu():
    print("============================================")
    print(" Super Background Downloader Pro++")
    print("============================================")
    print("1. minecraft\n2. subway\n3. satisfying\n4. food")
    print("A. All categories\nQ. Quit")
    return input("\nEnter choice: ").strip().lower()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--auto",type=int,help="Auto mode: download N per category")
    args=parser.parse_args()
    cleanup_temp()
    ix=load_index(); meta=[]
    cats=list(CONFIG["categories"].keys())
    if args.auto:
        for c in cats: download(c,args.auto,ix,meta)
        META_PATH.write_text(json.dumps(meta,indent=2),encoding="utf-8")
        return
    while True:
        ch=menu()
        if ch=="q": break
        if ch=="a":
            c=int(input("How many per category? ") or "1")
            for cat in cats: download(cat,c,ix,meta)
        if ch in ["1","2","3","4"]:
            all_cats=["minecraft","subway","satisfying","food"]
            cat=all_cats[int(ch)-1]
            c=int(input("How many videos? ") or "1")
            download(cat,c,ix,meta)
    META_PATH.write_text(json.dumps(meta,indent=2),encoding="utf-8")

if __name__=="__main__": 
    import contextlib
    main()
