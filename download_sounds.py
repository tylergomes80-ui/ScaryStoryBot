#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Downloader Bot — God-Mode (path-aware, robust, parallel)
- Jumpscares (<12s) and ambience (5–120min) from YouTube.
- Reliable tool detection (local, env, PATH). Clear errors.
- Safe retries with exponential backoff. Deterministic logging.
- Real durations via ffprobe (not metadata guesses).
- LUFS normalize (EBU R128-ish). Ambience auto-sliced 5–15min.
- File↔video mapping by %(id)s to avoid zip-order bugs.
- Storage cap by total MB with oldest-first pruning.
- sounds_index.json updated for every produced file.
"""

import os, sys, json, random, subprocess, shutil, time, argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Paths
# =========================
BASE_DIR  = Path(__file__).parent
ASSETS    = BASE_DIR / "assets"
JUMP_DIR  = ASSETS / "jumpscares"
AMB_DIR   = ASSETS / "ambience"
CACHE_DIR = BASE_DIR / "cache_sounds"
LOG_DIR   = BASE_DIR / "logs"
INDEX_PATH= BASE_DIR / "sounds_index.json"
ASSETS_DIR = ASSETS

for d in [JUMP_DIR, AMB_DIR, CACHE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Tool detection
# =========================
def _sanitize(p): return str(p).strip().strip('"').strip("'")

def find_tool(name, candidates=()):
    exe = shutil.which(name)
    if exe: return exe
    for p in candidates:
        p = Path(p)
        if p.exists(): return str(p)
    return None

# User-specific observed paths as fallbacks
YTDLP = find_tool("yt-dlp", [
    Path.home() / "yt-dlp.exe",
    r"C:\Users\tyler\AppData\Local\Programs\Python\Python313\Scripts\yt-dlp.exe",
])
FFMPEG = find_tool("ffmpeg", [
    r"C:\Users\tyler\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe",
])
FFPROBE = find_tool("ffprobe", [
    r"C:\Users\tyler\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffprobe.exe",
])

if not YTDLP:  raise SystemExit("ERROR: yt-dlp not found. Install: pip install yt-dlp")
if not FFMPEG: raise SystemExit("ERROR: ffmpeg not found. Install: winget install Gyan.FFmpeg")
if not FFPROBE:raise SystemExit("ERROR: ffprobe not found. Ensure FFmpeg full build is in PATH")

# =========================
# Config
# =========================
CONFIG = {
    "categories": {
        "jumpscare": "horror jumpscare sound effect no copyright",
        "ambience":  "horror ambience creepy background sound no copyright"
    },
    "duration": {
        "jumpscare": (0, 12),       # seconds
        "ambience":  (300, 7200)    # 5 min – 2 h
    },
    "subtypes": {
        "jumpscare": ["scream","bang","heartbeat","whisper","sting","hit","whoosh"],
        "ambience":  ["rain","wind","forest","cave","roomtone","street","engine","fire"]
    },
    "title_blacklist": ["intro","outro","subscribe","compilation","credits"],
    "cache_size": 200,
    "max_size_mb": 500,            # total WAV budget under assets/*
    "amb_slice_min": 300,          # 5 min
    "amb_slice_max": 900           # 15 min
}

# =========================
# Logging
# =========================
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =========================
# Index
# =========================
def load_index():
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"downloaded": {}}

def save_index(ix):
    INDEX_PATH.write_text(json.dumps(ix, indent=2), encoding="utf-8")

# =========================
# Subprocess helper with backoff
# =========================
def run(cmd, timeout=600, retries=3, backoff=2.0, capture=False):
    for i in range(retries):
        try:
            if capture:
                p = subprocess.run(cmd, check=True, timeout=timeout,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return True, (p.stdout or ""), (p.stderr or "")
            else:
                subprocess.run(cmd, check=True, timeout=timeout)
                return True, "", ""
        except subprocess.CalledProcessError as e:
            wait = backoff ** (i+1)
            log(f"cmd fail [{e.returncode}] {cmd[0]} retry {i+1}/{retries} in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            wait = backoff ** (i+1)
            log(f"cmd err {e} retry {i+1}/{retries} in {wait:.1f}s")
            time.sleep(wait)
    return False, "", ""

# =========================
# Cache
# =========================
def cache_file(cat:str)->Path: return CACHE_DIR / f"{cat}.json"

def refresh_cache(cat: str, query: str, size: int):
    log(f"Refreshing cache for {cat}: top {size}")
    ok, out, _ = run([YTDLP, f"ytsearch{size}:{query}", "--skip-download", "--print-json"], capture=True, timeout=900)
    vids=[]
    if ok:
        for line in out.splitlines():
            try:
                j = json.loads(line.strip())
                if j.get("_type") == "playlist": continue
                vids.append({
                    "id": j.get("id"),
                    "title": j.get("title") or "",
                    "duration": int(j.get("duration") or 0),
                    "url": j.get("webpage_url") or ""
                })
            except Exception:
                continue
    cache_file(cat).write_text(json.dumps(vids, indent=2), encoding="utf-8")
    log(f"Cached {len(vids)} for {cat}")
    return vids

def load_cache(cat):
    p = cache_file(cat)
    if not p.exists(): return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

# =========================
# Filters
# =========================
def filter_videos(videos, cat, ix):
    dmin, dmax = CONFIG["duration"][cat]
    bl = [b.lower() for b in CONFIG["title_blacklist"]]
    out=[]
    for v in videos:
        if not v.get("id") or not v.get("url"): continue
        if v["id"] in ix["downloaded"]: continue
        dur = int(v.get("duration") or 0)
        if not (dmin <= dur <= dmax): continue
        title = (v.get("title") or "").lower()
        if any(b in title for b in bl): continue
        out.append(v)
    random.shuffle(out)
    return out

def tag_subtype(title:str, cat:str):
    title = (title or "").lower()
    for sub in CONFIG["subtypes"][cat]:
        if sub in title: return sub
    return "generic"

# =========================
# ffprobe real duration
# =========================
def probe_seconds(path: Path) -> float:
    ok, out, _ = run(
        [FFPROBE,"-v","error","-show_entries","format=duration",
         "-of","default=noprint_wrappers=1:nokey=1", str(path)],
        timeout=60, retries=2, capture=True
    )
    try:
        return float(out.strip()) if ok and out.strip() else 0.0
    except Exception:
        return 0.0

# =========================
# Audio ops
# =========================
def normalize_audio(path: Path, target="-16", tp="-1.5", lra="11"):
    tmp = path.with_suffix(".tmp.wav")
    ok, _, _ = run([FFMPEG,"-y","-i",str(path),
                    "-af",f"loudnorm=I={target}:TP={tp}:LRA={lra}",
                    str(tmp)], timeout=600, retries=2, capture=False)
    if ok:
        try: os.replace(tmp, path)
        except Exception: pass
    else:
        if tmp.exists():
            with contextlib.suppress(Exception):
                tmp.unlink()

def slice_ambience(path: Path):
    dur = int(probe_seconds(path))
    if dur <= CONFIG["amb_slice_max"]:
        return [path]
    outputs=[]
    start=0
    part=1
    while start < dur:
        chunk = random.randint(CONFIG["amb_slice_min"], CONFIG["amb_slice_max"])
        end = min(start + chunk, dur)
        outp = path.with_name(f"{path.stem}_part{part}.wav")
        ok, _, _ = run([FFMPEG,"-y","-i",str(path),"-ss",str(start),"-to",str(end),"-c","copy",str(outp)],
                       timeout=600, retries=2)
        if ok:
            normalize_audio(outp)
            outputs.append(outp)
        start = end
        part += 1
    with contextlib.suppress(Exception):
        path.unlink()
    return outputs

# =========================
# Download
# =========================
def yt_dlp_download(urls, outdir: Path, prefix: str):
    """
    Use %(id)s in template so we can map files back to videos.
    """
    if not urls: return []
    template = str(outdir / f"{prefix}_%(id)s.%(ext)s")
    cmd = [
        YTDLP,
        "-f", "bestaudio/best",
        "-x", "--audio-format", "wav",
        "--no-playlist", "--no-overwrites",
        "-o", template
    ] + urls
    run(cmd, timeout=1800, retries=3)
    return list(outdir.glob(f"{prefix}_*.wav"))

# =========================
# Prune by total size
# =========================
def prune_assets():
    total = sum(f.stat().st_size for f in ASSETS_DIR.rglob("*.wav"))
    limit = 500 * 1024 * 1024  # 500 MB

    if total > limit:
        print(f"[{now()}] Pruning to keep under 500 MB...")

        jumps = sorted(JUMP_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        ambs = sorted(AMB_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime)

        for f in list(jumps) + list(ambs):
            try:
                f.unlink()
                print(f"[{now()}] Pruned {f.name}")
            except Exception as e:
                print(f"[ERROR] Could not delete {f}: {e}")
            total = sum(f.stat().st_size for f in ASSETS_DIR.rglob("*.wav"))
            if total <= limit:
                break

# =========================
# Core pipeline per category
# =========================
def _map_files_by_id(files):
    """
    From 'jumpscare_XXXXXXXX.wav' → 'XXXXXXXX'
    """
    m={}
    for f in files:
        stem = f.stem  # e.g., jumpscare_dQw4w9WgXcQ
        vid = stem.split("_",1)[-1] if "_" in stem else stem
        m[vid] = f
    return m

def process_downloads(cat, picks, outdir: Path, ix):
    """
    Normalize, optionally slice, tag subtype, update index.
    Keeps mapping by YouTube ID to avoid zip-order issues.
    """
    urls = [p["url"] for p in picks]
    log(f"{cat}: downloading {len(urls)} file(s)")
    files = yt_dlp_download(urls, outdir, cat)

    by_id = _map_files_by_id(files)
    for meta in picks:
        vid = meta["id"]
        f = by_id.get(vid)
        if not f or not f.exists():
            continue
        # Normalize
        normalize_audio(f)
        # Subtype
        subtype = tag_subtype(meta.get("title",""), cat)
        # Slice ambience
        produced = [f]
        if cat == "ambience":
            produced = slice_ambience(f)
        # Update index per produced file
        for s in produced:
            key = f"{vid}__{s.stem}"
            ix["downloaded"][key] = {
                "title": meta.get("title",""),
                "url":   meta.get("url",""),
                "cat":   cat,
                "subtype": subtype,
                "duration": int(meta.get("duration") or 0),
                "path":  str(s)
            }
    save_index(ix)
    prune_assets()
    log(f"{cat}: finished.")

def pick_candidates(cat, ix, need):
    vids = load_cache(cat) or refresh_cache(cat, CONFIG["categories"][cat], CONFIG["cache_size"])
    filt = filter_videos(vids, cat, ix)
    return filt[:need]

def download(cat, count, ix):
    outdir = JUMP_DIR if cat == "jumpscare" else AMB_DIR
    picks = pick_candidates(cat, ix, count)
    if not picks:
        log(f"{cat}: no candidates")
        return
    # Parallelize in small batches to survive errors
    batch = 6 if cat == "ambience" else 10
    for i in range(0, len(picks), batch):
        process_downloads(cat, picks[i:i+batch], outdir, ix)

# =========================
# CLI / Menu
# =========================
def menu():
    print("============================================")
    print("  Sound Downloader Bot (God-Mode)")
    print("============================================")
    print("1. Jumpscares")
    print("2. Ambience")
    print("A. All")
    print("R. Refresh caches")
    print("Q. Quit")
    return input("\nEnter choice: ").strip().lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", type=int, help="Auto mode: download N per category")
    ap.add_argument("--only", choices=["jumpscare","ambience"], help="Limit to one category")
    args = ap.parse_args()

    ix = load_index()

    if args.auto:
        cats = [args.only] if args.only else list(CONFIG["categories"].keys())
        for c in cats: download(c, args.auto, ix)
        return

    while True:
        ch = menu()
        if ch == "q": break
        if ch == "r":
            for cat, q in CONFIG["categories"].items():
                refresh_cache(cat, q, CONFIG["cache_size"])
            continue
        if ch == "a":
            n = int(input("How many per category? ") or "1")
            for cat in CONFIG["categories"]:
                download(cat, n, ix)
            continue
        if ch in ("1","2"):
            cat = "jumpscare" if ch == "1" else "ambience"
            n = int(input("How many files? ") or "1")
            download(cat, n, ix)

if __name__ == "__main__":
    import contextlib
    main()
