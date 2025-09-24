#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scary Story Bot — GOD-TIER main.py (monolithic)
All features inline. No external modules required beyond optional deps.
Parts: 1/6 header+imports+globals, 2/6 text+reddit+policy, 3/6 TTS,
4/6 media helpers, 5/6 rendering pipeline, 6/6 CLI+entrypoint.
"""

# =========================
# Part 1/6 — Header, imports, config, logging, utilities
# =========================
import os, re, json, time, glob, logging, random, asyncio, hashlib, argparse, math, contextlib, subprocess, shutil, sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Third-party libs (graceful fallbacks where possible)
try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, TextClip,
        CompositeVideoClip, ColorClip, concatenate_audioclips, afx
    )
    from moviepy.video.fx.all import loop
except Exception as e:
    print("\033[91m[ERROR]\033[0m moviepy not available. Install: pip install moviepy")
    raise

try:
    from moviepy.audio.AudioClip import AudioArrayClip
except Exception:
    AudioArrayClip = None  # handled later

try:
    import numpy as np
except Exception as e:
    print("\033[91m[ERROR]\033[0m numpy not available. Install: pip install numpy")
    raise

try:
    from pydub import AudioSegment
except Exception as e:
    print("\033[91m[ERROR]\033[0m pydub not available. Install: pip install pydub")
    raise

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional libs
try:
    from langdetect import detect as _lang_detect
except Exception:
    _lang_detect = None

try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _st_model = None

# ------------- Heartbeat + colored logs -------------
import threading

def heartbeat(label="ScaryStoryBot", interval=5):
    def loop():
        while True:
            sys.stdout.write(f"\r[{time.strftime('%H:%M:%S')}] {label} ... alive")
            sys.stdout.flush()
            time.sleep(interval)
    t = threading.Thread(target=loop, daemon=True)
    t.start()

def ok(msg):   print(f"\033[92m[OK]\033[0m {msg}")
def warn(msg): print(f"\033[93m[WARN]\033[0m {msg}")
def err(msg):  print(f"\033[91m[ERROR]\033[0m {msg}")

heartbeat("ScaryStoryBot")

# ------------- Logging -------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------------- Config dataclass -------------
@dataclass
class Config:
    out_dir: str = "output"
    tmp_dir: str = "tmp"
    assets_dir: str = "assets"
    bg_dir: str = "assets/backgrounds"
    music_dir: str = "assets/music"
    sfx_dir: str = "assets/sfx/jumpscares"
    overlay_dir: str = "assets/overlays"
    runlog_path: str = "output/runlog.jsonl"
    summary_md: str = "output/summary.md"

    subreddits: List[str] = field(default_factory=lambda: ["nosleep","shortscarystories","creepypasta"])
    timeframe: str = "month"
    limit: int = 50
    min_score: int = 15
    max_title_len: int = 160

    narrator_voice: str = os.getenv("NARRATOR_VOICE","en-US-ChristopherNeural")
    character_voice: str = os.getenv("CHAR_VOICE","en-US-JennyNeural")
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    sentence_pause_ms: int = 180

    resolution: Tuple[int,int] = (1080,1920)
    fps: int = 30
    text_margin: int = 80
    title_fontsize: int = 64
    caption_fontsize: int = 44
    part_badge_fontsize: int = 54
    fonts: Dict[str,str] = field(default_factory=lambda: {"title":"Impact","badge":"Impact","caption":"Arial-Bold","dyslexia":"OpenDyslexic"})
    max_part_seconds: int = 210
    min_part_seconds: int = 120
    max_parts: int = 4

    music_volume_db: float = -18.0
    sfx_volume_db: float = -4.0
    voice_normalize_target: float = -1.0

    allow_jumpscares: bool = True
    max_jumpscares_per_part: int = 3
    jumpscare_flash_ms: int = 140
    jumpscare_zoom_scale: float = 1.08
    jumpscare_candidates_topk: int = 10

    karaoke: bool = True
    dyslexia_friendly: bool = False
    apply_grain: bool = True
    apply_lut: bool = False

    # Visual polish
    watermark_path: Optional[str] = "assets/watermark.png"  # set None to disable
    watermark_opacity: float = 0.85
    watermark_secs: Tuple[float,float] = (0.0, 5.0)

    # Policy screens
    block_nsfw: bool = True
    block_hate: bool = True

    # Publishing metadata stubs
    hashtag_pool: List[str] = field(default_factory=lambda: ["#scarystory","#creepypasta","#horror","#nosleep","#shorts","#tiktok"])

    jumpscare_keywords: Dict[str,List[str]] = field(default_factory=lambda:{
        "generic":["suddenly","scream","screamed","behind me","door slammed","it grabbed","it moved","eyes opened",
                   "it smiled","footsteps","it was there","it wasn't there","went cold","I froze","silence",
                   "the lights","darkness","breathing","blood","ran","it stared","it turned"],
        "forest":["trees","woods","branches","trail","leaves","howl","cabin"],
        "room":["closet","bed","door","window","mirror","wardrobe"],
        "diner":["counter","kitchen","booth","cook","waitress"],
        "road":["highway","road","gas station","mile marker","hitchhiker","truck","tow truck"]
    })
    bg_keyword_map: Dict[str,str] = field(default_factory=lambda:{
        "forest":"forest","woods":"forest","room":"room","bedroom":"room",
        "diner":"diner","road":"road","highway":"road","gas":"road"
    })

    # Resilience
    min_free_gb: float = 2.0  # stop if less than this on output drive
    encoder_preference: List[str] = field(default_factory=lambda: ["h264_nvenc","h264_amf","h264_qsv","libx264"])
    encoder_bench_seconds: int = 4  # quick probe render seconds

cfg = Config()

# Deterministic randomness
random.seed(int(os.getenv("SEED", "42")))

# Metrics
METRICS = {"stories":0, "parts":0, "errors":0, "render_seconds":0.0}

# ------------- FS helpers -------------
def ensure_dirs():
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.runlog_path).parent).mkdir(parents=True, exist_ok=True)

def purge_tmp(older_than_hours=8):
    now=time.time()
    for p in Path(cfg.tmp_dir).glob("*"):
        with contextlib.suppress(Exception):
            if p.is_file() and now - p.stat().st_mtime > older_than_hours*3600:
                p.unlink()

def disk_free_gb(path: str) -> float:
    try:
        total, used, free = shutil.disk_usage(path)
        return free / (1024**3)
    except Exception:
        return 999.0  # assume fine if unknown

# ------------- Runlog + summary -------------
def runlog(event:dict):
    event["ts"] = datetime.utcnow().isoformat()+"Z"
    with open(cfg.runlog_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def append_summary(line: str):
    Path(cfg.summary_md).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.summary_md, "a", encoding="utf-8") as f:
        f.write(line.rstrip()+"\n")

# ------------- String helpers -------------
def clean_filename(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:80] if len(s) > 80 else s

def out_slug(post:dict) -> str:
    base = clean_filename(post.get("title") or "story")
    pid = post.get("id") or hashlib.sha1((post.get("url") or base).encode()).hexdigest()[:8]
    return f"{base}__{pid}"

# ------------- Audio helpers -------------
def read_duration_sec(audio_path: str) -> float:
    return AudioSegment.from_file(audio_path).duration_seconds

def normalize_audio_dbfs(audio: AudioSegment, target_db: float = -1.0) -> AudioSegment:
    gain = target_db - audio.max_dBFS
    return audio.apply_gain(gain)

def safe_get(obj,key,default=None):
    if isinstance(obj, dict): return obj.get(key, default)
    return getattr(obj,key,default)

def load_assets(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))

# ------------- Encoder detection / benchmarking -------------
def _encoders_supported() -> Dict[str,bool]:
    # Probe ffmpeg encoders list
    try:
        out = subprocess.check_output(["ffmpeg","-hide_banner","-encoders"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return {k: False for k in cfg.encoder_preference}
    support = {}
    for enc in cfg.encoder_preference:
        support[enc] = (enc in out)
    return support

ENC_SUPPORT = _encoders_supported()

def _pick_best_encoder() -> Tuple[str,str]:
    # Prefer available encoders in preferred order
    for enc in cfg.encoder_preference:
        if ENC_SUPPORT.get(enc):
            # presets per encoder family
            if enc == "libx264":
                return enc, "ultrafast"
            elif enc.startswith("h264_"):
                return enc, "p5"  # NVENC/AMF/QSV preset mid-high quality
    return "libx264", "ultrafast"

PREFERRED_ENCODER, PREFERRED_PRESET = _pick_best_encoder()
ok(f"Encoder selected: {PREFERRED_ENCODER} (preset {PREFERRED_PRESET})")

def _bench_encoder() -> Tuple[str,str]:
    """Quick 4s bench to confirm encoder works. Falls back if failure."""
    enc, preset = PREFERRED_ENCODER, PREFERRED_PRESET
    tmp_video = str(Path(cfg.tmp_dir) / "bench_src.mp4")
    tmp_out   = str(Path(cfg.tmp_dir) / "bench_out.mp4")
    try:
        # Generate a tiny color clip via moviepy to file
        clip = ColorClip(cfg.resolution, color=(10,10,10)).set_duration(cfg.encoder_bench_seconds)
        clip.write_videofile(tmp_video, fps=cfg.fps, codec="libx264", preset="ultrafast",
                             audio=False, verbose=False, logger=None)
        clip.close()
        # Try encoding with preferred encoder using ffmpeg CLI pass-through
        cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",tmp_video,
               "-c:v", enc, "-preset", preset, "-t", str(cfg.encoder_bench_seconds), tmp_out]
        res = subprocess.run(cmd, check=True)
        ok("Encoder bench passed")
        return enc, preset
    except Exception as e:
        warn(f"Encoder bench failed with {enc}: {e}. Falling back to libx264/ultrafast.")
        return "libx264","ultrafast"
    finally:
        with contextlib.suppress(Exception): os.remove(tmp_video)
        with contextlib.suppress(Exception): os.remove(tmp_out)

BENCH_ENCODER, BENCH_PRESET = _bench_encoder()
# =========================
# Part 2/6 — Reddit intake, policy, language, splitting, duration, dedupe
# =========================

# ---------- Reddit Fetch ----------
def fetch_posts(subs: List[str], timeframe: str, limit:int, min_score:int) -> List[dict]:
    items=[]
    try:
        import praw
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT","ScaryStoryBot/3.0 by u/yourbot")
        )
        for sub in subs:
            logging.info(f"Fetching r/{sub} top:{timeframe} limit:{limit}")
            for p in reddit.subreddit(sub).top(time_filter=timeframe, limit=limit):
                post={"title":safe_get(p,"title",""),
                      "selftext":safe_get(p,"selftext",""),
                      "score":safe_get(p,"score",0),
                      "subreddit":sub,
                      "id":safe_get(p,"id",""),
                      "url":f"https://reddit.com{safe_get(p,'permalink','')}",
                      "over_18":safe_get(p,"over_18",False)}
                if post["score"]>=min_score and (post["selftext"] or sub=="shortscarystories"):
                    items.append(post)
    except Exception as e:
        logging.warning(f"PRAW fetch failed: {e}")
        local_json = Path("posts.json")
        if local_json.exists():
            try:
                logging.info("Using local posts.json")
                items = json.loads(local_json.read_text(encoding="utf-8"))
            except Exception as e2:
                logging.error(f"Failed reading posts.json: {e2}")

    # Deduplicate using SimHash
    uniq={}
    seen_hashes=set()
    for it in items:
        key = it.get("id") or it.get("url") or it.get("title")
        text=(it.get("title") or "")+"\n"+(it.get("selftext") or "")
        h=_simhash(text)
        if any(_hamdist(h, sh) <= 3 for sh in seen_hashes):
            continue
        if key and key not in uniq:
            it["title"] = (it.get("title") or "")[:cfg.max_title_len]
            uniq[key] = it
            seen_hashes.add(h)
    logging.info(f"Candidates after filter: {len(uniq)}")
    return list(uniq.values())

# ---------- Policy screens ----------
_NSFW_RE = re.compile(r"\b(nsfw|sex|porn|nud(e|ity)|explicit)\b", re.I)
_HATE_RE = re.compile(r"\b(kill (them|all)|racial slur|gas (them|the)|nazi|genocide)\b", re.I)

def passes_policy(post:dict) -> bool:
    if post.get("over_18", False) and cfg.block_nsfw:
        return False
    txt = ((post.get("title") or "") + " \n " + (post.get("selftext") or "")).lower()
    if cfg.block_hate and _HATE_RE.search(txt):
        return False
    if cfg.block_nsfw and _NSFW_RE.search(txt):
        return False
    return True

# ---------- Language + text processing ----------
SENT_SPLIT_RE = re.compile(r"(?<=[\.?\!…])\s+(?=[A-Z0-9\"“‘])")

def to_story_text(post: dict) -> Optional[str]:
    title = (post.get("title") or "").strip()
    body = (post.get("selftext") or "").strip()
    return "\n\n".join([p for p in [title,body] if p]) if title or body else None

def detect_language(text:str) -> Optional[str]:
    if not _lang_detect:
        return None
    try:
        return _lang_detect(text)
    except Exception:
        return None

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+"," ", text).strip()
    if not text: return []
    sents = SENT_SPLIT_RE.split(text)
    merged, buf=[], ""
    for s in sents:
        s=s.strip()
        if not s: continue
        if len(s)<20 and buf: buf+=" "+s
        else:
            if buf: merged.append(buf.strip())
            buf=s
    if buf: merged.append(buf.strip())
    return merged

# ---------- Duration estimate + filtering ----------
def est_secs(text:str, wpm=155)->int:
    words = max(1, len(re.findall(r"\w+", text)))
    pauses = max(0, len(split_sentences(text)) - 1) * (cfg.sentence_pause_ms/1000)
    return int((words / wpm) * 60 + pauses)

def filter_posts(posts: List[dict]) -> List[dict]:
    out=[]
    for p in posts:
        if not passes_policy(p):
            continue
        t = to_story_text(p) or ""
        secs = est_secs(t)
        if cfg.min_part_seconds <= secs <= cfg.max_part_seconds and p.get("score",0) >= cfg.min_score:
            out.append(p)
    return out

# ---------- SimHash helpers ----------
def _simhash(text:str, bits=64) -> int:
    tokens=re.findall(r"\w+", text.lower())
    v=[0]*bits
    for tok in tokens:
        h=int(hashlib.sha1(tok.encode()).hexdigest(),16)
        for i in range(bits):
            v[i]+= 1 if (h>>i)&1 else -1
    out=0
    for i in range(bits):
        if v[i]>0: out|=(1<<i)
    return out

def _hamdist(a:int,b:int)->int:
    return bin(a^b).count("1")
# =========================
# Part 3/6 — TTS subsystem (Azure + Edge) with retries and cache
# =========================

# ---------- TTS ----------
USE_AZURE = bool(os.getenv("AZURE_SPEECH_KEY")) and bool(os.getenv("AZURE_SPEECH_REGION"))

if os.name == "nt":
    with contextlib.suppress(Exception):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def _edge_say(text, voice, out_path, rate, volume):
    import edge_tts
    await edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume).save(out_path)

async def _azure_say(text, voice, out_path, rate, volume):
    import azure.cognitiveservices.speech as speechsdk
    cfg_ = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
    cfg_.speech_synthesis_voice_name = voice
    ssml = f"""
    <speak version='1.0' xml:lang='en-US'>
      <voice name='{voice}'><prosody rate='{rate}' volume='{volume}'>{text}</prosody></voice>
    </speak>"""
    audio_cfg = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg_, audio_config=audio_cfg)
    res = synth.speak_ssml_async(ssml).get()
    if res.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(str(res.reason))

async def _tts_with_retry(text, voice, out_path, rate, volume, attempts=4):
    last=None
    for i in range(attempts):
        try:
            if USE_AZURE:
                await _azure_say(text, voice, out_path, rate, volume)
            else:
                await _edge_say(text, voice, out_path, rate, volume)
            return
        except Exception as e:
            last=e
            await asyncio.sleep(0.6*(2**i)+random.random()*0.2)
    raise last

async def edge_tts_speak(text:str,voice:str,out_path:str,rate:str="+0%",volume:str="+0%"):
    await _tts_with_retry(text, voice, out_path, rate, volume)

# Cache helpers
def tts_cache_path(text:str, voice:str)->str:
    h = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()
    return f"{cfg.tmp_dir}/tts_{voice}_{h}.mp3"

async def synth_sentence(text, voice):
    out_path = tts_cache_path(text, voice)
    if os.path.exists(out_path):
        return out_path, read_duration_sec(out_path)
    try:
        await edge_tts_speak(text, voice, out_path, cfg.tts_rate, cfg.tts_volume)
    except Exception:
        dur_ms = max(800, min(5000, int(len(text)*50)))
        AudioSegment.silent(dur_ms).export(out_path, format="mp3")
    return out_path, read_duration_sec(out_path)

async def synth_all_sentences(sentences:List[str], voice:str) -> List[Tuple[Optional[str],float]]:
    sem = asyncio.Semaphore(8)
    async def worker(s):
        async with sem:
            return await synth_sentence(s, voice)
    base = await asyncio.gather(*(worker(s) for s in sentences))
    # interleave virtual gaps (None path)
    out=[]
    for i,(p,d) in enumerate(base,1):
        out.append((p,d))
        if cfg.sentence_pause_ms>0 and i < len(base):
            out.append((None, cfg.sentence_pause_ms/1000))
    return out
# =========================
# Part 4/6 — Backgrounds, Jumpscares, Audio Mixing, Captions
# =========================

# ---------- Jumpscare ----------
SUDDEN_RE = re.compile(r"\b(suddenly|all at once|without warning)\b", re.I)
SAFE_RE = re.compile(r"\b(calm|quiet|nothing happened|finally safe)\b", re.I)

def detect_bg_flavor(bg_path: Optional[str]) -> str:
    if not bg_path: return "generic"
    low = Path(bg_path).stem.lower()
    for kw, flavor in cfg.bg_keyword_map.items():
        if kw in low: return flavor
    return "generic"

def sentence_intensity(s: str, flavor: str) -> float:
    s_low = s.lower()
    score = s.count("!")*1.2 + (0.6 if "..." in s or "…" in s else 0) + min(1.0,len(s)/140.0)
    for w in cfg.jumpscare_keywords.get("generic", []):
        if w in s_low: score+=1.0
    for w in cfg.jumpscare_keywords.get(flavor, []):
        if w in s_low: score+=1.0
    if SUDDEN_RE.search(s_low): score+=1.5
    if SAFE_RE.search(s_low): score-=0.8
    return max(0.0, score)

def choose_jumpscare_positions(sentences: List[str], flavor: str, max_sc: int) -> List[int]:
    cand=[(i,sentence_intensity(s,flavor)) for i,s in enumerate(sentences)]
    cand=[(i,sc) for i,sc in cand if i>=2]  # avoid cold open
    cand.sort(key=lambda x:x[1],reverse=True)
    chosen=[]
    for i,sc in cand[:cfg.jumpscare_candidates_topk]:
        if sc<=2.2: break
        if not any(abs(i-c)<3 for c in chosen):
            chosen.append(i)
        if len(chosen)>=max_sc: break
    return sorted(chosen)

# ---------- Video/Audio helpers ----------
def _nvenc_available()->bool:
    try:
        out = subprocess.check_output(["ffmpeg","-hide_banner","-encoders"], stderr=subprocess.STDOUT, text=True)
        return "h264_nvenc" in out
    except Exception:
        return False

NVENC_OK = _nvenc_available()

def loudnorm_wav(in_wav: str) -> str:
    out = in_wav.replace(".wav", "_ln.wav")
    try:
        subprocess.run(["ffmpeg","-y","-i", in_wav,
                        "-af","loudnorm=I=-14:TP=-1.0:LRA=11:print_format=summary",
                        out], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except Exception:
        return in_wav

def load_looping_bg(total_dur: float, bg_paths: List[str]) -> Tuple[VideoFileClip, Optional[str]]:
    if not bg_paths:
        return ColorClip(cfg.resolution, color=random.choice([(10,10,10),(20,0,0),(0,20,0),(0,0,30)])).set_duration(total_dur).without_audio(), None
    chosen = _select_bg(total_dur, bg_paths)
    try:
        clip=VideoFileClip(chosen).without_audio().fx(afx.volumex,0.0).resize(cfg.resolution)
        # Add subtle Ken Burns
        clip = _ken_burns(clip, total_dur)
        # Grain and LUT
        if cfg.apply_grain:
            clip = _add_grain(clip, intensity=0.04)
        if cfg.apply_lut:
            clip = _tone_map(clip)
        return loop(clip, duration=total_dur+0.2), chosen
    except Exception as e:
        logging.warning(f"BG load failed ({chosen}): {e}")
        return ColorClip(cfg.resolution, color=(10,10,10)).set_duration(total_dur).without_audio(), None

def _ken_burns(clip:VideoFileClip, duration:float) -> VideoFileClip:
    # Simple zoom-in
    scale_start, scale_end = 1.02, 1.08
    def make_frame(t):
        prog = min(1.0, t/duration)
        scale = scale_start + (scale_end-scale_start)*prog
        frame = clip.get_frame(t)
        # resize via numpy center crop
        h,w = frame.shape[0], frame.shape[1]
        nh, nw = int(h/scale), int(w/scale)
        y0 = (h-nh)//2; x0 = (w-nw)//2
        return frame[y0:y0+nh, x0:x0+nw]
    return clip.fl(make_frame)

def _add_grain(clip:VideoFileClip, intensity=0.03) -> VideoFileClip:
    rng = np.random.default_rng(123)
    def fl(frame):
        noise = rng.normal(0, 255*intensity, size=frame.shape).astype(np.int16)
        out = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return out
    return clip.fl_image(fl)

def _tone_map(clip:VideoFileClip) -> VideoFileClip:
    # Simple gamma adjustment to emulate LUT
    gamma = 0.9
    inv = 1.0/gamma
    lut = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype(np.uint8)
    def fl(frame):
        return lut[frame]
    return clip.fl_image(fl)

def _select_bg(total_dur:float, bg_paths:List[str]) -> str:
    if _st_model:
        # choose randomly for now; hook point for text-conditioned bg
        return random.choice(bg_paths)
    return random.choice(bg_paths)
# =========================
# Part 5/6 — Audio Mixing, Captions, Splitting, Thumbnails
# =========================

def _concat_with_silence(items: List[Tuple[Optional[str],float]]):
    sr=44100
    clips=[]
    t_local=0.0
    sentence_timings_local=[]
    for p,d in items:
        if p is None:
            n=int(d*sr)
            clips.append(AudioArrayClip(np.zeros((n,1), dtype=np.float32), fps=sr))
        else:
            ac=AudioFileClip(p)
            clips.append(ac)
            sentence_timings_local.append((t_local, t_local+ac.duration))
        t_local += d
    concat=concatenate_audioclips(clips)
    raw=f"{cfg.tmp_dir}/voice_part_raw_{int(time.time()*1000)}.wav"
    concat.write_audiofile(raw, fps=sr, codec="pcm_s16le", verbose=False, logger=None)
    for c in clips:
        with contextlib.suppress(Exception): c.close()
    seg=AudioSegment.from_file(raw)
    seg_norm=normalize_audio_dbfs(seg, cfg.voice_normalize_target)
    out=f"{cfg.tmp_dir}/voice_part_{int(time.time()*1000)}.wav"
    seg_norm.export(out, format="wav")
    return out, sentence_timings_local, t_local

def pick_music(music_paths: List[str]) -> Optional[AudioSegment]:
    if not music_paths: return None
    path=random.choice(music_paths)
    try:
        return AudioSegment.from_file(path).apply_gain(cfg.music_volume_db)
    except Exception as e:
        logging.warning(f"Music load failed: {e}")
        return None

def make_music_bed(duration_sec: float, music_paths: List[str]) -> Optional[str]:
    mseg = pick_music(music_paths)
    if mseg is None: return None
    loops=[]
    filled=0
    target_ms = int(duration_sec*1000) + 1200
    while filled < target_ms:
        loops.append(mseg)
        filled += len(mseg)
    bed=sum(loops).fade_in(1200).fade_out(1600)
    out=f"{cfg.tmp_dir}/music_{int(time.time()*1000)}.wav"
    bed[:int(duration_sec*1000)].export(out,format="wav")
    return out

def _rms_envelope(wav_path:str, frame_ms=120) -> np.ndarray:
    snd = AudioSegment.from_file(wav_path)
    samples = np.array(snd.get_array_of_samples()).astype(np.float32)
    if snd.channels>1:
        samples = samples.reshape((-1, snd.channels)).mean(axis=1)
    sr = snd.frame_rate
    hop = int(sr*frame_ms/1000)
    env=[]
    for i in range(0, len(samples), hop):
        seg = samples[i:i+hop]
        if len(seg)==0: break
        env.append(float(np.sqrt((seg**2).mean()+1e-9)))
    arr = np.array(env)
    if arr.size:
        arr = arr/ (arr.max()+1e-9)
    return arr

def duck(bed: AudioSegment, base_reduction_db=8) -> AudioSegment:
    return bed.apply_gain(-base_reduction_db)

def mix_audio_layers(voice_path: str, duration: float, music_path: Optional[str], sfx_events: List[Tuple[float,str]]) -> str:
    voice = AudioSegment.from_file(voice_path)
    total = voice
    if music_path:
        bed = AudioSegment.from_file(music_path)[:len(voice)]
        bed = duck(bed, base_reduction_db=6)
        env = _rms_envelope(voice_path)
        if env.size:
            frame_ms=120
            pieces=[]
            for i,val in enumerate(env):
                seg = bed[i*frame_ms:(i+1)*frame_ms]
                gain = -6*float(val)
                pieces.append(seg.apply_gain(gain))
            bed = sum(pieces)
        total = bed.overlay(total)
    for (at,p) in sfx_events:
        try:
            sfx = AudioSegment.from_file(p).apply_gain(cfg.sfx_volume_db)
            pre = max(0, int(at*1000)-200)
            post = min(len(total), pre+600)
            total = total[:pre] + total[pre:post].apply_gain(-5) + total[post:]
            total = total.overlay(sfx, position=max(0,int(at*1000)-int(cfg.jumpscare_flash_ms/2)))
        except Exception as e:
            logging.warning(f"SFX mix error: {e}")
    out = f"{cfg.tmp_dir}/mix_{int(time.time()*1000)}.wav"
    total.export(out, format="wav")
    out = loudnorm_wav(out)
    return out

# ---------- Captions ----------
def write_srt(path:str, sentences:List[str], timings:List[Tuple[float,float]]):
    def fmt(t):
        ms=int((t - int(t))*1000)
        h=int(t//3600); m=int((t%3600)//60); s=int(t%60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines=[]
    for i,(txt,(t0,t1)) in enumerate(zip(sentences,timings),1):
        lines.append(str(i))
        lines.append(f"{fmt(t0)} --> {fmt(t1)}")
        lines.append(txt.replace("\n"," ").strip())
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

def approx_word_timings(sentence:str, t0:float, t1:float) -> List[Tuple[str,float,float]]:
    words = [w for w in re.findall(r"\S+", sentence) if w]
    if not words:
        return []
    dur = max(0.01, t1-t0)
    step = dur/len(words)
    out=[]
    for i,w in enumerate(words):
        out.append((w, t0 + i*step, t0 + (i+1)*step))
    return out

def write_word_srt(path:str, sentences:List[str], timings:List[Tuple[float,float]]):
    idx=1
    def fmt(t):
        ms=int((t - int(t))*1000)
        h=int(t//3600); m=int((t%3600)//60); s=int(t%60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines=[]
    for s,(t0,t1) in zip(sentences, timings):
        for w, a, b in approx_word_timings(s, t0, t1):
            lines.append(str(idx)); idx+=1
            lines.append(f"{fmt(a)} --> {fmt(b)}")
            lines.append(w)
            lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

# ---------- Text overlays ----------
def safe_textclip(txt, **kw):
    if cfg.dyslexia_friendly:
        kw["font"] = cfg.fonts.get("dyslexia") or kw.get("font")
    fonts_try = [kw.pop("font", None), "Impact", "Arial-Bold", "DejaVu-Sans", None]
    for f in fonts_try:
        try:
            return TextClip(txt, method="caption", font=f, **kw)
        except Exception:
            pass
    return TextClip(txt, method="label")

def overlay_captions(base: VideoFileClip, sentences: List[str], timings: List[Tuple[float,float]],
                     part_badge: str, title: str) -> VideoFileClip:
    W,H=cfg.resolution
    overlays=[base]
    with contextlib.suppress(Exception):
        overlays.append(safe_textclip(title, fontsize=cfg.title_fontsize, color="white", stroke_color="black", stroke_width=2,
                                 size=(W-cfg.text_margin*2,None)).set_pos(("center",40)).set_duration(base.duration))
    with contextlib.suppress(Exception):
        overlays.append(safe_textclip(part_badge, fontsize=cfg.part_badge_fontsize, color="yellow", stroke_color="black", stroke_width=3)
                        .set_pos((W-cfg.text_margin-360,40)).set_duration(base.duration))
    y_pos=H-340
    if cfg.karaoke:
        for s,(t0,t1) in zip(sentences,timings):
            for w, a, b in approx_word_timings(s, t0, t1):
                with contextlib.suppress(Exception):
                    overlays.append(safe_textclip(w, fontsize=cfg.caption_fontsize, color="white", stroke_color="black", stroke_width=2,
                                             size=(W-cfg.text_margin*2,None))
                                      .set_pos(("center",y_pos)).set_start(a).set_duration(max(0.05,b-a)))
    else:
        for (s,(t0,t1)) in zip(sentences,timings):
            dur=max(0.1, t1-t0)
            with contextlib.suppress(Exception):
                overlays.append(safe_textclip(s, fontsize=cfg.caption_fontsize, color="white", stroke_color="black", stroke_width=2,
                                         size=(W-cfg.text_margin*2,None)).set_pos(("center",y_pos)).set_start(t0).set_duration(dur))
    return CompositeVideoClip(overlays, size=cfg.resolution)

def apply_jumpscare_effects(video: VideoFileClip, target_sentence_idxs: List[int],
                            sentence_timings: List[Tuple[float,float]]) -> VideoFileClip:
    if not target_sentence_idxs: return video
    overlays=[video]
    for idx in target_sentence_idxs:
        if idx>=len(sentence_timings): continue
        t0,t1 = sentence_timings[idx]
        center_t = t0 + min(0.9, (t1-t0)*0.35)
        start=max(0, center_t-(cfg.jumpscare_flash_ms/2000.0))
        end=min(video.duration-1.5, center_t+(cfg.jumpscare_flash_ms/2000.0))
        if end <= start: continue
        flash_img = os.path.join(cfg.overlay_dir,"jumpscare_flash.png")
        with contextlib.suppress(Exception):
            if os.path.exists(flash_img):
                overlays.append(ImageClip(flash_img).set_start(start).set_duration(end-start)
                                .resize(cfg.resolution).set_opacity(0.85))
            else:
                overlays.append(ColorClip(size=cfg.resolution, color=(250,250,250))
                                .set_start(start).set_duration(end-start).set_opacity(0.9))
        with contextlib.suppress(Exception):
            overlays.append(video.subclip(start,min(video.duration,start+0.6)).resize(cfg.jumpscare_zoom_scale).set_start(start))
    return CompositeVideoClip(overlays, size=cfg.resolution)

# ---------- Splitting ----------
def plan_parts_by_time(items: List[Tuple[Optional[str],float]]) -> List[Tuple[int,int]]:
    idxs=[]; t=0.0; start=0
    for i,(p,d) in enumerate(items):
        t += d
        is_gap = (p is None)
        if t >= cfg.max_part_seconds and len(idxs) < cfg.max_parts-1:
            end_i = i+1
            idxs.append((start, end_i))
            start = end_i
            t = 0.0
    idxs.append((start, len(items)))
    if len(idxs) >= 2:
        last = idxs[-1]
        last_dur = sum(items[i][1] for i in range(last[0], last[1]))
        if last_dur < cfg.min_part_seconds:
            prev = idxs[-2]
            idxs[-2] = (prev[0], last[1])
            idxs.pop()
    return idxs

# ---------- Thumbnails ----------
def save_thumbnail(clip:VideoFileClip, sentence_timings:List[Tuple[float,float]], js_idx:List[int], out_base:str):
    t = clip.duration/2
    if js_idx:
        target = js_idx[0]
        if 0<=target<len(sentence_timings):
            t = sum(sentence_timings[target])/2
    frame = clip.get_frame(t)
    from PIL import Image
    im = Image.fromarray(frame)
    thumb_path = f"{out_base}__thumb.jpg"
    im.save(thumb_path, quality=90)
    return thumb_path

def gen_hashtags() -> List[str]:
    return random.sample(cfg.hashtag_pool, k=min(4,len(cfg.hashtag_pool)))
# =========================
# Part 6/6 — Rendering pipeline + CLI
# =========================

def add_watermark(clip: VideoFileClip) -> VideoFileClip:
    p = cfg.watermark_path
    if not p or not os.path.exists(p):
        return clip
    try:
        wm = ImageClip(p).set_opacity(cfg.watermark_opacity).set_duration(clip.duration)
        # bottom-right with 20px margin
        W,H = cfg.resolution
        wm = wm.set_pos((W - wm.w - 20, H - wm.h - 20))
        return CompositeVideoClip([clip, wm], size=cfg.resolution)
    except Exception as e:
        logging.warning(f"Watermark failed: {e}")
        return clip

def process_post(post: dict, bg_paths: List[str], music_paths: List[str], dry_run=False) -> List[str]:
    title = post.get("title") or "(untitled)"
    story_text = to_story_text(post)
    if not story_text: raise ValueError("Empty story text")

    # Language detect
    lang = detect_language(story_text) or "en"
    if lang != "en":
        logging.info(f"Non-English detected ({lang}). Proceeding without translation.")

    sentences = split_sentences(story_text)
    if not sentences: raise ValueError("No sentences after split")
    logging.info(f"[TTS] Total sentences: {len(sentences)}")

    if dry_run:
        return []

    sent_audio = asyncio.run(synth_all_sentences(sentences, cfg.narrator_voice))

    parts = plan_parts_by_time(sent_audio)
    total_parts = len(parts)
    outputs = []

    sfx_list = sorted(glob.glob(os.path.join(cfg.sfx_dir,"*.mp3"))) + sorted(glob.glob(os.path.join(cfg.sfx_dir,"*.wav")))

    # Build prefix map from non-gap index
    non_gap_idx=[]
    for p,_ in sent_audio:
        non_gap_idx.append(0 if p is None else 1)
    non_gap_prefix=[0]
    c=0
    for v in non_gap_idx:
        if v==1: c+=1
        non_gap_prefix.append(c)

    slug = out_slug(post)

    # Disk guard
    if disk_free_gb(cfg.out_dir) < cfg.min_free_gb:
        raise RuntimeError(f"Insufficient disk space in {cfg.out_dir}. Need >= {cfg.min_free_gb} GB free.")

    for pi, (s0,s1) in enumerate(parts,1):
        part_items = sent_audio[s0:s1]
        if not part_items: continue

        # Rebuild per-part voice wav with in-memory silences
        voice_wav, sentence_timings_local, part_duration = _concat_with_silence(part_items)

        # Map to sentence texts using prefix sums
        start_ng = non_gap_prefix[s0]
        part_sentence_texts = sentences[start_ng:start_ng+len(sentence_timings_local)]

        # Background
        bg_video, chosen_bg = load_looping_bg(part_duration, bg_paths)
        bg_flavor = detect_bg_flavor(chosen_bg)

        # Jumpscares
        js_positions = []
        sfx_events=[]
        if cfg.allow_jumpscares:
            js_positions = choose_jumpscare_positions(part_sentence_texts, bg_flavor, cfg.max_jumpscares_per_part)
            for ridx in js_positions:
                if ridx < len(sentence_timings_local):
                    at = sentence_timings_local[ridx][0] + 0.1
                    if sfx_list:
                        sfx_events.append((at, sfx_list[(ridx*131 + pi*17) % len(sfx_list)]))

        # Music
        music_wav = make_music_bed(part_duration, music_paths)

        # Mix audio layers
        final_audio_wav = mix_audio_layers(voice_wav, part_duration, music_wav, sfx_events)

        # Visual effects and captions
        video_with_effects = apply_jumpscare_effects(bg_video, js_positions, sentence_timings_local)
        badge = f"Part {pi}/{total_parts}"
        titled = overlay_captions(video_with_effects, part_sentence_texts, sentence_timings_local, badge, title)
        watermarked = add_watermark(titled)
        final_video = watermarked.set_audio(AudioFileClip(final_audio_wav)).set_duration(part_duration)

        out_base = f"{cfg.out_dir}/{slug}__part_{pi}_of_{total_parts}"
        out_path = f"{out_base}.mp4"
        logging.info(f"Rendering: {out_path}")

        codec = BENCH_ENCODER
        preset = BENCH_PRESET

        with contextlib.suppress(Exception):
            write_srt(f"{out_base}.srt", part_sentence_texts, sentence_timings_local)
        with contextlib.suppress(Exception):
            write_word_srt(f"{out_base}.word.srt", part_sentence_texts, sentence_timings_local)

        t0=time.time()
        try:
            final_video.write_videofile(out_path, fps=cfg.fps, codec=codec, preset=preset,
                                        audio_codec="aac", threads=os.cpu_count() or 4,
                                        temp_audiofile=f"{cfg.tmp_dir}/_aud_{pi}.m4a",
                                        remove_temp=True, verbose=False, logger=None)
        finally:
            METRICS["render_seconds"] += time.time()-t0
            # Cleanup video objects
            with contextlib.suppress(Exception):
                if hasattr(final_video, 'audio') and isinstance(final_video.audio, AudioFileClip):
                    final_video.audio.close()
            with contextlib.suppress(Exception): final_video.close()
            with contextlib.suppress(Exception): watermarked.close()
            with contextlib.suppress(Exception): titled.close()
            with contextlib.suppress(Exception): video_with_effects.close()
            with contextlib.suppress(Exception): bg_video.close()

        # Thumbnail
        with contextlib.suppress(Exception):
            thumb = save_thumbnail(VideoFileClip(out_path), sentence_timings_local, js_positions, out_base)

        # Metadata
        meta = {
            "title": title,
            "subreddit": post.get("subreddit"),
            "reddit_url": post.get("url"),
            "post_id": post.get("id"),
            "part_index": pi,
            "part_count": total_parts,
            "duration_sec": round(part_duration,2),
            "bg_video": Path(chosen_bg).name if chosen_bg else None,
            "bg_flavor": bg_flavor,
            "jumpscare_sentences": js_positions,
            "voices": {"narrator": cfg.narrator_voice, "character": cfg.character_voice},
            "hashtags": random.sample(cfg.hashtag_pool, k=min(4,len(cfg.hashtag_pool))),
            "rendered_at": datetime.utcnow().isoformat()+"Z",
            "encoder": {"codec": codec, "preset": preset}
        }
        Path(f"{out_base}._meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        append_summary(f"- {out_path}  |  {meta['duration_sec']}s  |  r/{meta['subreddit']}  |  {meta['reddit_url']}")
        outputs.append(out_path)
        METRICS["parts"] += 1

    METRICS["stories"] += 1
    return outputs

# ---------- CLI / entrypoint ----------
def require_env(keys:List[str]):
    missing=[k for k in keys if not os.getenv(k)]
    if missing:
        logging.warning(f"Missing env: {missing}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subs", type=str, default=",".join(cfg.subreddits), help="Comma-separated subreddit list")
    parser.add_argument("--timeframe", type=str, default=cfg.timeframe, help="hour, day, week, month, year, all")
    parser.add_argument("--limit", type=int, default=cfg.limit)
    parser.add_argument("--minscore", type=int, default=cfg.min_score)
    parser.add_argument("--prefilter", action="store_true", help="Filter stories to ~3–5 min before TTS")
    parser.add_argument("--maxvideos", type=int, default=0, help="Stop after N rendered parts total (0 = no cap)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel posts")
    parser.add_argument("--dryrun", action="store_true", help="Plan only. No TTS/render.")
    parser.add_argument("--interactive", action="store_true", help="Confirm each candidate before render")
    args = parser.parse_args()

    cfg.subreddits = [s.strip() for s in args.subs.split(",") if s.strip()]
    cfg.timeframe = args.timeframe
    cfg.limit = args.limit
    cfg.min_score = args.minscore

    ensure_dirs(); purge_tmp()
    require_env(["REDDIT_CLIENT_ID","REDDIT_CLIENT_SECRET","REDDIT_USER_AGENT"])
    if USE_AZURE:
        require_env(["AZURE_SPEECH_KEY","AZURE_SPEECH_REGION"])

    logging.info(f"=== Scary Story Bot Starting ===")
    logging.info(f"Encoder: {BENCH_ENCODER} preset={BENCH_PRESET}")
    logging.info(f"Voices: narrator={cfg.narrator_voice} character={cfg.character_voice}")

    posts = [p for p in fetch_posts(cfg.subreddits, cfg.timeframe, cfg.limit, cfg.min_score) if passes_policy(p)]
    if args.prefilter:
        posts = filter_posts(posts)
    if not posts:
        logging.info("No posts fetched. Using fallback test story.")
        posts=[{
            "title":"The Knock at the Window",
            "selftext":"Last night, I woke up to a knock at my window... but I live on the 14th floor.",
            "score":9999,
            "subreddit":"test",
            "id":"fallback001",
            "url":"https://example.com"
        }]

    if args.interactive:
        kept=[]
        for p in posts:
            print(f"\n[r/{p.get('subreddit')}] score={p.get('score')}\n{p.get('title')}\nProceed? [y/N] ", end="")
            ans=input().strip().lower()
            if ans=="y": kept.append(p)
        posts=kept or posts

    bg_paths = load_assets(os.path.join(cfg.bg_dir,"*.mp4"))
    music_paths = load_assets(os.path.join(cfg.music_dir,"*.mp3")) + load_assets(os.path.join(cfg.music_dir,"*.wav"))

    total_generated = 0

    def _run_one(post):
        try:
            score = int(post.get("score",0))
            logging.info(f"Processing: r/{post.get('subreddit')} | score={score} | {post.get('title')}")
            outputs = process_post(post, bg_paths, music_paths, dry_run=args.dryrun)
            if outputs:
                logging.info(f"SUCCESS: {len(outputs)} file(s): {outputs}")
            else:
                logging.info("No outputs for this post.")
            runlog({"event":"post_done","post_id":post.get("id"),"parts":len(outputs)})
            return outputs
        except Exception as e:
            METRICS["errors"] += 1
            logging.error(f"Skipping story due to error: {e}", exc_info=True)
            runlog({"event":"post_error","post_id":post.get("id"),"error":str(e)})
            return []

    if args.workers>1 and not args.dryrun:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs=[ex.submit(_run_one, p) for p in posts]
            for f in as_completed(futs):
                outs=f.result()
                total_generated += len(outs)
                if args.maxvideos and total_generated >= args.maxvideos:
                    break
    else:
        for post in posts:
            outs=_run_one(post)
            total_generated += len(outs)
            if args.maxvideos and total_generated >= args.maxvideos:
                break

    runlog({"event":"run_done","stories":METRICS["stories"],"parts":METRICS["parts"],"errors":METRICS["errors"],"render_s":round(METRICS['render_seconds'],2)})
    logging.info(f"=== Done. Generated {total_generated} file(s). Output dir: {cfg.out_dir} ===")
    print("\nBot finished. Check output folder for video.")

if __name__=="__main__":
    main()





