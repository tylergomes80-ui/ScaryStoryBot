# main.py
# Scary Story Bot Extreme — Advanced, with intelligent jumpscares and multi-part splitting
# Dependencies (pip): moviepy, pydub, edge-tts (optional), praw (optional), python-dotenv (optional)
# Assets expected:
#   ./assets/backgrounds/*.mp4 (loopable ambience videos; include keywords in filenames like "forest","room","diner","road")
#   ./assets/music/*.mp3           (ambient loops)
#   ./assets/sfx/jumpscares/*.mp3  (different categories; e.g., slam.mp3, shriek.mp3, boom.mp3)
#   ./assets/overlays/jumpscare_flash.png (optional white flash)
# Config via env or defaults below.

import os
import re
import json
import math
import time
import glob
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    ImageClip,
    TextClip,
    CompositeVideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
    afx,
)
from pydub import AudioSegment

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ---------- Config ----------

@dataclass
class Config:
    # IO
    out_dir: str = "output"
    tmp_dir: str = "tmp"
    assets_dir: str = "assets"
    bg_dir: str = "assets/backgrounds"
    music_dir: str = "assets/music"
    sfx_dir: str = "assets/sfx/jumpscares"
    overlay_dir: str = "assets/overlays"

    # Fetch
    subreddits: List[str] = field(default_factory=lambda: ["nosleep", "shortscarystories", "creepypasta"])
    timeframe: str = "month"      # day|week|month|year|all
    limit: int = 50
    min_score: int = 450
    max_title_len: int = 160

    # TTS
    narrator_voice: str = os.getenv("NARRATOR_VOICE", "en-US-ChristopherNeural")
    character_voice: str = os.getenv("CHAR_VOICE", "en-US-JennyNeural")
    tts_rate: str = "+0%"          # for edge-tts if used
    tts_volume: str = "+0%"        # for edge-tts if used
    sentence_pause_ms: int = 180   # natural gap between sentences

    # Video
    resolution: Tuple[int, int] = (1080, 1920)  # 9:16 TikTok
    fps: int = 30
    text_margin: int = 80
    title_fontsize: int = 64
    caption_fontsize: int = 44
    part_badge_fontsize: int = 54
    fonts: Dict[str, str] = field(default_factory=lambda: {
        "title": "Impact",
        "badge": "Impact",
        "caption": "Arial-Bold"
    })
    max_part_seconds: int = 210     # ~3.5min target per part
    min_part_seconds: int = 120     # don’t create micro parts below this if avoidable
    max_parts: int = 4

    # Audio mix
    music_volume_db: float = -18.0
    sfx_volume_db: float = -4.0
    voice_normalize_target: float = -1.0  # dBFS target peak normalize

    # Jumpscare
    allow_jumpscares: bool = True
    max_jumpscares_per_part: int = 3
    jumpscare_flash_ms: int = 140
    jumpscare_zoom_scale: float = 1.08
    jumpscare_candidates_topk: int = 10
    jumpscare_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "generic": ["suddenly", "scream", "screamed", "behind me", "door slammed", "it grabbed", "it moved",
                    "eyes opened", "it smiled", "footsteps", "it was there", "it wasn't there", "went cold", "I froze",
                    "silence", "the lights", "darkness", "breathing", "blood", "ran", "it stared", "it turned"],
        "forest": ["trees", "woods", "branches", "trail", "leaves", "howl", "cabin"],
        "room": ["closet", "bed", "door", "window", "mirror", "wardrobe"],
        "diner": ["counter", "kitchen", "booth", "cook", "waitress"],
        "road": ["highway", "road", "gas station", "mile marker", "hitchhiker", "truck", "tow truck"]
    })
    # Match background filename keywords to a jumpscare flavor
    bg_keyword_map: Dict[str, str] = field(default_factory=lambda: {
        "forest": "forest",
        "woods": "forest",
        "room": "room",
        "bedroom": "room",
        "diner": "diner",
        "road": "road",
        "highway": "road",
        "gas": "road"
    })

cfg = Config()

# ---------- Utils ----------

def ensure_dirs():
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tmp_dir).mkdir(parents=True, exist_ok=True)

def clean_filename(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:80] if len(s) > 80 else s

def read_duration_sec(audio_path: str) -> float:
    seg = AudioSegment.from_file(audio_path)
    return seg.duration_seconds

def normalize_audio_dbfs(in_path: str, target_peak_dbfs: float = -1.0) -> str:
    seg = AudioSegment.from_file(in_path)
    peak = seg.max_dBFS
    gain = target_peak_dbfs - peak
    normalized = seg.apply_gain(gain)
    out_path = in_path.replace(".mp3", "_norm.mp3")
    normalized.export(out_path, format="mp3")
    return out_path

def safe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def load_assets(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    return files

# ---------- Reddit Fetch (optional PRAW) ----------

def fetch_posts(subs: List[str], timeframe: str, limit: int, min_score: int) -> List[dict]:
    """
    Returns list of dicts with keys: title, selftext, score, subreddit, id, url
    Uses PRAW if creds present; otherwise returns empty (you can inject posts.json yourself).
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "ScaryStoryBot/1.0 by u/yourbot")
    items: List[dict] = []

    if client_id and client_secret:
        try:
            import praw  # type: ignore
            reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
            for sub in subs:
                logging.info(f"Fetching r/{sub} top:{timeframe} limit:{limit}")
                sr = reddit.subreddit(sub)
                for p in sr.top(time_filter=timeframe, limit=limit):
                    post = {
                        "title": safe_get(p, "title", ""),
                        "selftext": safe_get(p, "selftext", ""),
                        "score": safe_get(p, "score", 0),
                        "subreddit": sub,
                        "id": safe_get(p, "id", ""),
                        "url": f"https://reddit.com{safe_get(p, 'permalink', '')}",
                        "over_18": safe_get(p, "over_18", False)
                    }
                    if post["score"] >= min_score and (post["selftext"] or sub == "shortscarystories"):
                        items.append(post)
        except Exception as e:
            logging.warning(f"PRAW fetch failed: {e}. Returning empty set.")
    else:
        # Fallback: try local cached posts.json if present
        local_json = Path("posts.json")
        if local_json.exists():
            logging.info("Using local posts.json")
            try:
                items = json.loads(local_json.read_text())
            except Exception as e:
                logging.error(f"Failed reading posts.json: {e}")

    # Deduplicate by id/url and truncate titles
    uniq = {}
    for it in items:
        key = it.get("id") or it.get("url")
        if not key:
            key = it.get("title")
        if key and key not in uniq:
            it["title"] = (it["title"] or "")[:cfg.max_title_len]
            uniq[key] = it
    output = list(uniq.values())
    logging.info(f"Candidates after filter: {len(output)}")
    return output

# ---------- Text Processing ----------

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!…])\s+(?=[A-Z0-9\"“‘])")

def to_story_text(post: dict) -> Optional[str]:
    title = (post.get("title") or "").strip()
    body = (post.get("selftext") or "").strip()
    if not title and not body:
        return None
    # For shortscarystories, often only title + short body; keep both
    parts = [p for p in [title, body] if p]
    return "\n\n".join(parts)

def split_sentences(text: str) -> List[str]:
    # Normalize weird whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Guard for empty
    if not text:
        return []
    # Split
    sents = SENT_SPLIT_RE.split(text)
    # Merge too-short fragments
    merged: List[str] = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) < 20 and buf:
            buf += " " + s
        else:
            if buf:
                merged.append(buf.strip())
            buf = s
    if buf:
        merged.append(buf.strip())
    return merged

# ---------- TTS ----------

async def edge_tts_speak(text: str, voice: str, out_path: str, rate: str = "+0%", volume: str = "+0%"):
    import edge_tts  # type: ignore
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume)
    await communicate.save(out_path)

def synth_tts_sentence(text: str, voice: str, idx: int, base_name: str) -> Tuple[str, float]:
    """
    Synthesize a single sentence to mp3. Tries edge-tts; if unavailable, falls back to pydub tone (debug).
    Returns (path, duration_sec)
    """
    safe = clean_filename(base_name)
    out_path = f"{cfg.tmp_dir}/{safe}_s{idx:04d}.mp3"
    try:
        # Prefer edge-tts (async) but run via a local event loop for simplicity
        import asyncio
        asyncio.run(edge_tts_speak(text, voice, out_path, cfg.tts_rate, cfg.tts_volume))
    except Exception as e:
        logging.warning(f"TTS fallback for sentence {idx}: {e}")
        # Fallback: simple tone placeholder so pipeline still works
        dur_ms = max(800, min(5000, int(len(text) * 50)))
        tone = AudioSegment.silent(duration=dur_ms)
        tone.export(out_path, format="mp3")
    # Normalize
    try:
        out_path = normalize_audio_dbfs(out_path, cfg.voice_normalize_target)
    except Exception as e:
        logging.warning(f"Normalize failed: {e}")
    dur = read_duration_sec(out_path)
    return out_path, dur

def tts_sentences(sentences: List[str], voice: str, base_name: str) -> List[Tuple[str, float]]:
    clips = []
    for i, s in enumerate(sentences, 1):
        logging.info(f"[TTS] Sentence {i}/{len(sentences)} using {voice}: {s[:60]}{'...' if len(s)>60 else ''}")
        path, dur = synth_tts_sentence(s, voice, i, base_name)
        clips.append((path, dur))
        # Insert small pause between sentences
        if cfg.sentence_pause_ms > 0:
            pause = AudioSegment.silent(duration=cfg.sentence_pause_ms)
            p = f"{cfg.tmp_dir}/{clean_filename(base_name)}_gap_{i:04d}.mp3"
            pause.export(p, format="mp3")
            clips.append((p, pause.duration_seconds))
    return clips

# ---------- Scoring for Jumpscares ----------

def detect_bg_flavor(bg_path: str) -> str:
    low = Path(bg_path).stem.lower()
    for kw, flavor in cfg.bg_keyword_map.items():
        if kw in low:
            return flavor
    return "generic"

def sentence_intensity(s: str, flavor: str) -> float:
    s_low = s.lower()
    score = 0.0
    # punctuation / length
    score += s.count("!") * 1.2
    score += 0.6 if "..." in s or "…" in s else 0.0
    score += min(1.0, len(s) / 140.0)

    # generic keywords
    for w in cfg.jumpscare_keywords["generic"]:
        if w in s_low:
            score += 1.0

    # flavor keywords
    if flavor in cfg.jumpscare_keywords:
        for w in cfg.jumpscare_keywords[flavor]:
            if w in s_low:
                score += 1.0

    # abrupt tokens
    if re.search(r"\b(suddenly|all at once|without warning)\b", s_low):
        score += 1.5

    # negative calmers
    if re.search(r"\b(calm|quiet|nothing happened|finally safe)\b", s_low):
        score -= 0.8

    return max(0.0, score)

def choose_jumpscare_positions(sentences: List[str], bg_flavor: str, max_sc: int) -> List[int]:
    scores = [(i, sentence_intensity(s, bg_flavor)) for i, s in enumerate(sentences)]
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = []
    for i, sc in scores[: cfg.jumpscare_candidates_topk]:
        # avoid clustering — keep distance of at least 3 sentences
        if not any(abs(i - c) < 3 for c in chosen) and sc > 2.2:
            chosen.append(i)
        if len(chosen) >= max_sc:
            break
    return sorted(chosen)

# ---------- Audio/Video Build ----------

def build_voice_audio(sentence_audio: List[Tuple[str, float]]) -> Tuple[str, List[Tuple[float, float]]]:
    """
    Concatenate sentence MP3s; return single mp3 path + per-sentence timing [(start,end)]
    """
    clips = [AudioFileClip(p) for (p, _) in sentence_audio]
    concat = concatenate_audioclips(clips)
    out_mp3 = f"{cfg.tmp_dir}/voice_{int(time.time())}.mp3"
    concat.write_audiofile(out_mp3, fps=44100, verbose=False, logger=None)
    # Build timings
    timings = []
    t = 0.0
    for (p, d) in sentence_audio:
        timings.append((t, t + d))
        t += d
    for c in clips:
        c.close()
    return out_mp3, timings

def load_looping_bg(total_dur: float, bg_paths: List[str]) -> Tuple[VideoFileClip, str]:
    """
    Loop/concatenate background videos to cover total duration.
    Returns (video_clip, chosen_bg_path)
    """
    if not bg_paths:
        raise RuntimeError("No background videos found in assets/backgrounds")
    chosen = random.choice(bg_paths)
    base = VideoFileClip(chosen).without_audio().fx(afx.volumex, 0.0)
    base = base.resize(cfg.resolution)
    clips = []
    filled = 0.0
    while filled < total_dur + 2:
        clips.append(base.copy())
        filled += base.duration
    video = concatenate_videoclips(clips).set_duration(total_dur + 0.2)
    return video, chosen

def pick_music(music_paths: List[str]) -> Optional[AudioSegment]:
    if not music_paths:
        return None
    path = random.choice(music_paths)
    try:
        seg = AudioSegment.from_file(path).apply_gain(cfg.music_volume_db)
        return seg
    except Exception as e:
        logging.warning(f"Music load failed: {e}")
        return None

def make_music_bed(duration_sec: float, music_paths: List[str]) -> Optional[str]:
    mseg = pick_music(music_paths)
    if mseg is None:
        return None
    loops = []
    filled = 0.0
    while filled < duration_sec * 1000 + 1000:
        loops.append(mseg)
        filled += len(mseg)
    bed = sum(loops).fade_in(1200).fade_out(1600)
    out = f"{cfg.tmp_dir}/music_{int(time.time())}.mp3"
    bed[: int(duration_sec * 1000)].export(out, format="mp3")
    return out

def overlay_captions(base: VideoFileClip, sentences: List[str], timings: List[Tuple[float, float]],
                     part_badge: str, title: str) -> VideoFileClip:
    W, H = cfg.resolution
    overlays = []

    # Title (top)
    try:
        title_clip = TextClip(title, fontsize=cfg.title_fontsize, font=cfg.fonts["title"],
                              color="white", stroke_color="black", stroke_width=2, method="caption",
                              size=(W - cfg.text_margin * 2, None,)).set_pos(("center", 40)).set_duration(base.duration)
        overlays.append(title_clip)
    except Exception:
        pass

    # Part badge (top-right)
    try:
        badge = TextClip(part_badge, fontsize=cfg.part_badge_fontsize, font=cfg.fonts["badge"],
                         color="yellow", stroke_color="black", stroke_width=3).set_pos((W - cfg.text_margin - 360, 40)).set_duration(base.duration)
        overlays.append(badge)
    except Exception:
        pass

    # Timed captions (bottom)
    y_pos = H - 340
    for (s, (t0, t1)) in zip(sentences, timings):
        dur = max(0.1, t1 - t0)
        try:
            cap = TextClip(s, fontsize=cfg.caption_fontsize, font=cfg.fonts["caption"],
                           color="white", stroke_color="black", stroke_width=2, method="caption",
                           size=(W - cfg.text_margin * 2, None,)).set_pos(("center", y_pos)).set_start(t0).set_duration(dur)
            overlays.append(cap)
        except Exception:
            # If font missing, skip captions rather than crash
            pass

    return CompositeVideoClip([base, *overlays], size=cfg.resolution)

def apply_jumpscare_effects(video: VideoFileClip, timings: List[Tuple[float, float]],
                            target_sentence_idxs: List[int], sentence_timings: List[Tuple[float, float]],
                            sfx_paths: List[str]) -> VideoFileClip:
    """
    Add flash + subtle zoom + SFX centered near the start of chosen sentences
    """
    if not target_sentence_idxs:
        return video

    overlays = [video]
    for idx in target_sentence_idxs:
        if idx >= len(sentence_timings):
            continue
        t0, t1 = sentence_timings[idx]
        center_t = t0 + min(0.9, (t1 - t0) * 0.35)
        start = max(0, center_t - (cfg.jumpscare_flash_ms / 2000.0))
        end = min(video.duration, center_t + (cfg.jumpscare_flash_ms / 2000.0))

        # Flash overlay
        flash_img = os.path.join(cfg.overlay_dir, "jumpscare_flash.png")
        if os.path.exists(flash_img):
            flash = (ImageClip(flash_img)
                     .set_start(start)
                     .set_duration(end - start)
                     .resize(cfg.resolution)
                     .set_opacity(0.85))
            overlays.append(flash)
        else:
            # fallback: white text block
            try:
                white = (TextClip(" ", fontsize=1, color="white", bg_color="white", size=cfg.resolution)
                         .set_start(start).set_duration(end - start).set_opacity(0.9))
                overlays.append(white)
            except Exception:
                pass

        # Zoom effect by resizing and cropping slightly (simulate shock)
        try:
            zoomed = (video.subclip(start, min(video.duration, start + 0.6))
                      .resize(cfg.jumpscare_zoom_scale))
            # Place zoomed clip over base (centered)
            zoomed = zoomed.set_start(start)
            overlays.append(zoomed)
        except Exception:
            pass

        # SFX
        try:
            if sfx_paths:
                sfx = random.choice(sfx_paths)
                s = AudioSegment.from_file(sfx).apply_gain(cfg.sfx_volume_db)
                out = f"{cfg.tmp_dir}/sfx_{idx:03d}.mp3"
                s.export(out, format="mp3")
                # Add onto existing audio later during audio mix
                # We don’t attach here at the video layer; we’ll mix below
        except Exception as e:
            logging.warning(f"SFX inject failed: {e}")

    return CompositeVideoClip(overlays, size=cfg.resolution)

def mix_audio_layers(voice_mp3: str, duration: float, music_mp3: Optional[str], sfx_events: List[Tuple[float, str]]) -> str:
    """
    Mix voice, music bed and timed sfx_events [(at_sec, path)]
    """
    voice = AudioSegment.from_file(voice_mp3)
    total = voice

    if music_mp3:
        bed = AudioSegment.from_file(music_mp3)
        bed = bed[:len(voice)]
        total = bed.overlay(total)

    for (at, p) in sfx_events:
        try:
            sfx = AudioSegment.from_file(p).apply_gain(cfg.sfx_volume_db)
            at_ms = int(at * 1000)
            total = total.overlay(sfx, position=max(0, at_ms - int(cfg.jumpscare_flash_ms / 2)))
        except Exception as e:
            logging.warning(f"SFX mix error: {e}")

    out = f"{cfg.tmp_dir}/mix_{int(time.time())}.mp3"
    total.export(out, format="mp3")
    return out

# ---------- Part Splitting ----------

def plan_parts(sentence_audio: List[Tuple[str, float]]) -> List[Tuple[int, int]]:
    """
    Given per-sentence audio durations (with gaps included), decide part cut indices.
    Returns list of (start_idx, end_idx_exclusive) per part.
    """
    durs = [d for (_, d) in sentence_audio]
    cum = 0.0
    cuts = [0]
    for i, d in enumerate(durs):
        cum += d
        if cum >= cfg.max_part_seconds and len(cuts) < cfg.max_parts:
            cuts.append(i + 1)
            cum = 0.0
    if cuts[-1] != len(durs):
        cuts.append(len(durs))

    # Merge tiny tail if below min_part_seconds
    parts = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    if len(parts) >= 2:
        last_dur = sum(durs[parts[-1][0]:parts[-1][1]])
        if last_dur < cfg.min_part_seconds:
            a0, a1 = parts[-2]
            parts[-2] = (a0, parts[-1][1])
            parts.pop()
    return parts

# ---------- Main Processing ----------

def process_post(post: dict, bg_paths: List[str], music_paths: List[str], sfx_paths: List[str]) -> List[str]:
    title = post.get("title") or "(untitled)"
    story_text = to_story_text(post)
    if not story_text:
        raise ValueError("Empty story text")
    sentences = split_sentences(story_text)
    if not sentences:
        raise ValueError("No sentences after split")

    logging.info(f"[TTS] Total sentences: {len(sentences)}")

    # Narrate all with narrator voice for now (simple/clean); you can add character switching if you tag quotes.
    sent_audio = tts_sentences(sentences, cfg.narrator_voice, base_name=title)

    # Plan parts based on durations
    parts = plan_parts(sent_audio)
    total_parts = len(parts)
    outputs: List[str] = []

    # Preload assets
    bg_flavor = "generic"  # fallback; will set per-part based on chosen bg
    sfx_list = sorted(glob.glob(os.path.join(cfg.sfx_dir, "*.mp3")))

    # Build each part
    for pi, (s0, s1) in enumerate(parts, 1):
        part_sentences = sentences[s0:s1]
        part_audio_list = sent_audio[s0:s1]
        if not part_sentences or not part_audio_list:
            continue

        # Voice audio + timings (for this part)
        voice_mp3, timings_all = build_voice_audio(part_audio_list)
        # timings_all includes gaps; but we only need sentence indices — filter to actual sentences positions
        # part_audio_list includes gaps after each sentence; we must extract positions of the sentence mp3s only.
        # Build sentence-only timings:
        t = 0.0
        sentence_timings = []
        for (path, dur) in part_audio_list:
            is_gap = "_gap_" in path
            if not is_gap:
                sentence_timings.append((t, t + dur))
            t += dur

        part_duration = t

        # Background
        bg_video, chosen_bg = load_looping_bg(part_duration, bg_paths)
        bg_flavor = detect_bg_flavor(chosen_bg)

        # Decide jumpscares
        js_positions_local = []
        sfx_events = []
        if cfg.allow_jumpscares:
            # choose sentence indexes relative to part
            rel_js_sent_idxs = choose_jumpscare_positions(part_sentences, bg_flavor, cfg.max_jumpscares_per_part)
            js_positions_local = rel_js_sent_idxs
            # Create SFX events at each target sentence time
            for ridx in rel_js_sent_idxs:
                if ridx < len(sentence_timings):
                    at = sentence_timings[ridx][0] + 0.1
                    # pick an sfx file deterministically by sentence index to avoid repetition patterns
                    if sfx_list:
                        sfx = sfx_list[(ridx * 131 + pi * 17) % len(sfx_list)]
                        sfx_events.append((at, sfx))

        # Music bed
        music_mp3 = make_music_bed(part_duration, music_paths)

        # Mix audio layers
        final_audio_mp3 = mix_audio_layers(voice_mp3, part_duration, music_mp3, sfx_events)

        # Apply jumpscare visual effects
        video_with_effects = apply_jumpscare_effects(bg_video, timings_all, js_positions_local, sentence_timings, sfx_list)

        # Overlay captions + labels
        badge = f"Part {pi}/{total_parts}"
        titled = overlay_captions(video_with_effects, part_sentences, sentence_timings, badge, title)

        # Attach final audio
        final_video = titled.set_audio(AudioFileClip(final_audio_mp3)).set_duration(part_duration)

        # Export
        slug = clean_filename(f"{title}") or f"story_{post.get('id','post')}"
        out_path = f"{cfg.out_dir}/{slug}__part_{pi}_of_{total_parts}.mp4"
        logging.info(f"Rendering: {out_path}")
        final_video.write_videofile(
            out_path,
            fps=cfg.fps,
            codec="libx264",
            audio_codec="aac",
            threads=os.cpu_count() or 4,
            temp_audiofile=f"{cfg.tmp_dir}/_aud_{pi}.m4a",
            remove_temp=True,
            verbose=False,
            logger=None,
        )

        # Cleanup moviepy objects
        try:
            final_video.close()
            titled.close()
            video_with_effects.close()
            bg_video.close()
        except Exception:
            pass

        # Write metadata
        meta = {
            "title": title,
            "subreddit": post.get("subreddit"),
            "reddit_url": post.get("url"),
            "post_id": post.get("id"),
            "part_index": pi,
            "part_count": total_parts,
            "duration_sec": round(part_duration, 2),
            "bg_video": Path(chosen_bg).name if chosen_bg else None,
            "bg_flavor": bg_flavor,
            "jumpscare_sentences": js_positions_local,
            "voices": {"narrator": cfg.narrator_voice, "character": cfg.character_voice},
            "rendered_at": datetime.utcnow().isoformat() + "Z",
        }
        Path(out_path.replace(".mp4", "._meta.json")).write_text(json.dumps(meta, indent=2))
        outputs.append(out_path)

    return outputs

# ---------- Main ----------

def main():
    ensure_dirs()
    logging.info("=== Scary Story Bot Extreme Starting ===")
    logging.info(f"Voices: narrator={cfg.narrator_voice} character={cfg.character_voice}")

    posts = fetch_posts(cfg.subreddits, cfg.timeframe, cfg.limit, cfg.min_score)
    if not posts:
        logging.info("No posts fetched. Exiting.")
        return

    # Assets
    bg_paths = load_assets(os.path.join(cfg.bg_dir, "*.mp4"))
    music_paths = load_assets(os.path.join(cfg.music_dir, "*.mp3"))
    sfx_paths = load_assets(os.path.join(cfg.sfx_dir, "*.mp3"))

    total_generated = 0
    for post in posts:
        try:
            score = int(post.get("score", 0))
            logging.info(f"Processing: r/{post.get('subreddit')} | score={score} | {post.get('title')}")
            outputs = process_post(post, bg_paths, music_paths, sfx_paths)
            if outputs:
                total_generated += len(outputs)
                logging.info(f"SUCCESS: {len(outputs)} file(s): {outputs}")
            else:
                logging.info("No outputs for this post (skipped or failed).")
        except Exception as e:
            logging.error(f"Skipping story due to error: {e}", exc_info=True)

    logging.info(f"=== Done. Generated {total_generated} file(s). Output dir: {cfg.out_dir} ===")
    print("\nBot finished. Check output folder for video.")

if __name__ == "__main__":
    main()
Paste your patch below. Press CTRL+Z then Enter when finished:
Paste your patch below. Press CTRL+Z then Enter when finished:
