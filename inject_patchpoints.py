#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Injector script: adds PATCHPOINT anchors to main.py
- Makes a backup as main_backup.py
- Inserts anchors before relevant functions
"""

import re
from pathlib import Path

MAIN_PATH = Path("main.py")
BACKUP_PATH = Path("main_backup.py")

PATCHPOINTS = {
    "reddit_fetch": r"def\s+fetch_reddit",
    "story_split": r"def\s+split_story",
    "tts_setup": r"def\s+setup_tts",
    "background_select": r"def\s+select_background",
    "sfx_insert": r"def\s+insert_jumpscare|def\s+add_sfx",
    "final_render": r"def\s+render_final|def\s+export_video"
}

def inject_patchpoints():
    if not MAIN_PATH.exists():
        print("❌ main.py not found in this folder.")
        return

    # backup first
    BACKUP_PATH.write_text(MAIN_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    text = MAIN_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_lines = []
    added = set()

    for line in lines:
        # check for each patchpoint
        inserted = False
        for key, pattern in PATCHPOINTS.items():
            if re.search(pattern, line) and key not in added:
                out_lines.append(f"# PATCHPOINT: {key}")
                added.add(key)
                inserted = True
                break
        out_lines.append(line)

    MAIN_PATH.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"✅ PATCHPOINTs inserted into {MAIN_PATH}, backup at {BACKUP_PATH}")

if __name__ == "__main__":
    inject_patchpoints()
Paste your patch below. Press CTRL+Z then Enter when finished:
Paste your patch below. Press CTRL+Z then Enter when finished:
