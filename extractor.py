#!/usr/bin/env python3
"""
Binary audio container extractor for Angela/Ginger/Tom .bin files.

Confirmed structure (from Audacity + byte analysis):
  0x0000–0x5650  : Header + seek-point table (4499 x 4-byte LE ROM addresses, base 0x10000000)
  0x5650–SONG_START : ADPCM voice clips (batch 1) – boundaries from seek table
  SONG_START–SONG_END : PCM songs, signed 8-bit, 8 kHz (raw linear stream)
  SONG_END–EOF   : ADPCM voice clips (batch 2) – boundaries from sync-pattern runs

Sync pattern (silence/gap marker): 00 08 c6 01 00 ca 00 c0  (8 bytes, repeated ≥3x = clip boundary)

Songs: decode directly as signed 8-bit PCM at 8000 Hz.
Voice clips: decode as Dialogic VOX ADPCM at 8000 Hz; brute-force
  alignment shifts 0–16 bytes × {normal, nibble-swapped} and keep the
  candidate with lowest spectral flatness (most tonal, least noise-like).
"""

import os, sys, struct, subprocess, argparse
import numpy as np
from pathlib import Path

# ── per-file constants ─────────────────────────────────────────────────────────
FILE_CFG = {
    "Tom":    {"song_start": 0x34E00, "song_end": 0x10D800},
    "Angela": {"song_start": 0x35000, "song_end": 0x10D000},
    "Ginger": {"song_start": 0x35000, "song_end": 0x10D000},
}

BASE_ADDR   = 0x10000000
TABLE_START = 0x1004
SYNC_PAT    = bytes([0x00, 0x08, 0xc6, 0x01, 0x00, 0xca, 0x00, 0xc0])
MIN_SYNC_RUN = 3   # ≥3 consecutive 8-byte patterns = real gap between clips
MIN_CLIP_BYTES = 256


# ── helpers ────────────────────────────────────────────────────────────────────

def read_seek_table(data):
    """Return list of all (file_offset) values from the seek table."""
    offsets = []
    i = TABLE_START
    while i + 4 <= len(data):
        val = struct.unpack_from("<I", data, i)[0]
        if (val >> 24) == 0x10:
            offsets.append(val - BASE_ADDR)
            i += 4
        else:
            break
    return offsets


def get_presong_clips(data, song_start):
    """Return [(start, end)] for voice clips in 0x5650–song_start using seek table."""
    all_entries = read_seek_table(data)
    frames = sorted(set(all_entries[12:]))          # skip 12-entry track directory
    voice = sorted(e for e in frames if 0x5000 <= e < song_start)

    clips = []
    for j, off in enumerate(voice):
        end = voice[j + 1] if j + 1 < len(voice) else song_start
        if end - off >= MIN_CLIP_BYTES:
            clips.append((off, end))
    return clips


def find_sync_runs(data, region_start, region_end):
    """Find all runs of ≥MIN_SYNC_RUN consecutive 8-byte sync patterns."""
    runs = []
    i = region_start
    while i + 8 <= region_end:
        if data[i:i+8] == SYNC_PAT:
            run_start = i
            while i + 8 <= region_end and data[i:i+8] == SYNC_PAT:
                i += 8
            run_len = (i - run_start) // 8
            if run_len >= MIN_SYNC_RUN:
                runs.append((run_start, i))   # (gap_start, gap_end)
        else:
            i += 8
    return runs


def get_postsong_clips(data, song_end):
    """Return [(start, end)] for voice clips in song_end–EOF using sync-run boundaries."""
    region_end = len(data)
    runs = find_sync_runs(data, song_end, region_end)
    if not runs:
        return []

    clips = []
    # Audio starts after first sync run
    prev_end = runs[0][1]
    for (gs, ge) in runs[1:]:
        start, end = prev_end, gs
        if end - start >= MIN_CLIP_BYTES:
            clips.append((start, end))
        prev_end = ge
    # Remainder after last sync run
    if region_end - prev_end >= MIN_CLIP_BYTES:
        clips.append((prev_end, region_end))
    return clips


def decode_vox(binfile, outfile, rate=8000):
    """Decode Dialogic VOX ADPCM with sox."""
    r = subprocess.run(
        ["sox", "-t", "vox", "-r", str(rate), "-c", "1", str(binfile), str(outfile)],
        capture_output=True
    )
    return r.returncode == 0


def _swap_nibbles(data: bytes) -> bytes:
    """Swap the high and low nibble of every byte: (b>>4)|((b&0x0F)<<4)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return ((arr >> 4) | ((arr & 0x0F) << 4)).tobytes()


def find_best_shift(clip_data, out_dir, prefix, rate=8000, max_shift=16):
    """Search for the best-aligned VOX ADPCM decode across two nibble variants
    and byte shifts 0–max_shift.

    Candidates tested (2 × (max_shift+1) = 34 by default):
      - normal bytes,        shift 0–16
      - nibble-swapped bytes, shift 0–16

    Returns (shift, nibble_swap, final_wav_path, sfm, dur, clip_pct) for the
    candidate with the lowest spectral flatness (most tonal / least noise-like),
    or None if every attempt fails.  `nibble_swap` is a bool indicating whether
    the winning candidate used nibble-swapped input.  Only the winning WAV is kept.
    """
    best_info = None   # (shift, nibble_swap, sfm, dur, clip_pct)
    best_wav  = None   # Path of the current best WAV

    for nibble_swap in (False, True):
        base = _swap_nibbles(clip_data) if nibble_swap else clip_data
        tag  = "ns" if nibble_swap else "no"

        for shift in range(max_shift + 1):
            sliced = base[shift:]
            if len(sliced) < MIN_CLIP_BYTES:
                continue

            tmp_bin = out_dir / f"{prefix}_{tag}_s{shift:02d}.bin"
            tmp_wav = out_dir / f"{prefix}_{tag}_s{shift:02d}.wav"
            tmp_bin.write_bytes(sliced)

            ok = decode_vox(tmp_bin, tmp_wav, rate)
            tmp_bin.unlink(missing_ok=True)

            if not ok:
                tmp_wav.unlink(missing_ok=True)
                continue

            q = quality(tmp_wav)
            if q is None:
                tmp_wav.unlink(missing_ok=True)
                continue

            clip_pct, sfm, dur = q
            if best_info is None or sfm < best_info[2]:
                if best_wav is not None:
                    best_wav.unlink(missing_ok=True)
                best_info = (shift, nibble_swap, sfm, dur, clip_pct)
                best_wav  = tmp_wav
            else:
                tmp_wav.unlink(missing_ok=True)

    if best_info is None or best_wav is None:
        return None

    shift, nibble_swap, sfm, dur, clip_pct = best_info
    final_wav = out_dir / f"{prefix}_best_vox{rate}.wav"
    best_wav.rename(final_wav)
    return shift, nibble_swap, final_wav, sfm, dur, clip_pct


def decode_pcm(binfile, outfile, rate=8000):
    """Decode signed 8-bit PCM with sox."""
    r = subprocess.run(
        ["sox", "-t", "s8", "-r", str(rate), "-c", "1", str(binfile), str(outfile)],
        capture_output=True
    )
    return r.returncode == 0


def quality(wavfile_path):
    """Return (clip_pct, sfm) or None on failure."""
    try:
        from scipy.io import wavfile as wf
        from scipy import signal as sg
        rate, d = wf.read(str(wavfile_path))
        if d.dtype == np.int16:
            f = d.astype(float) / 32768.0
        else:
            f = d.astype(float) / 128.0
        clip_pct = np.mean(np.abs(f) >= 0.99) * 100
        freq, psd = sg.welch(f, rate, nperseg=min(256, max(16, len(f)//4)))
        psd_pos = psd[psd > 0]
        sfm = float(np.exp(np.mean(np.log(psd_pos))) / np.mean(psd_pos)) if len(psd_pos) else 1.0
        dur = len(d) / rate
        return clip_pct, sfm, dur
    except Exception:
        return None


# ── main ───────────────────────────────────────────────────────────────────────

def process_file(bin_path: Path, out_dir: Path, rates=(8000, 11025, 16000)):
    name = bin_path.stem          # "Tom", "Angela", "Ginger"
    cfg  = FILE_CFG.get(name)
    if cfg is None:
        # Auto-detect song boundaries
        data = bin_path.read_bytes()
        cfg = auto_detect_boundaries(data)
        print(f"[{name}] auto-detected: song_start=0x{cfg['song_start']:x} song_end=0x{cfg['song_end']:x}")
    else:
        data = bin_path.read_bytes()

    song_start = cfg["song_start"]
    song_end   = cfg["song_end"]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"File : {bin_path.name}  ({len(data)} bytes)")
    print(f"Songs: 0x{song_start:x}–0x{song_end:x}  ({(song_end-song_start)/8000:.1f}s PCM)")
    print(f"{'='*60}")

    # ── 1. Extract PCM songs ──────────────────────────────────────────────────
    song_data = data[song_start:song_end]
    song_bin  = out_dir / f"{name.lower()}_songs.bin"
    song_wav  = out_dir / f"{name.lower()}_songs_pcm8k.wav"
    song_bin.write_bytes(song_data)
    decode_pcm(song_bin, song_wav, rate=8000)
    print(f"[songs]  {len(song_data)} bytes → {song_wav.name}")

    # ── 2. Pre-song voice clips (ADPCM, seek-table boundaries) ───────────────
    pre_clips = get_presong_clips(data, song_start)
    print(f"\n[pre-song clips]  {len(pre_clips)} clips found via seek table")
    for j, (start, end) in enumerate(pre_clips):
        size = end - start
        raw = out_dir / f"{name.lower()}_pre_{j:03d}.bin"
        raw.write_bytes(data[start:end])
        result = find_best_shift(data[start:end], out_dir, f"{name.lower()}_pre_{j:03d}")
        if result:
            shift, nibble_swap, wav, sfm, dur, clip_pct = result
            ns_tag = " nibble-swap" if nibble_swap else ""
            print(f"  pre_{j:03d}  0x{start:07x} {size:7d}B  shift={shift:2d}B{ns_tag}  {wav.name}  {dur:.2f}s  clip={clip_pct:.0f}%  sfm={sfm:.3f}")
        else:
            print(f"  pre_{j:03d}  0x{start:07x} {size:7d}B  (decode failed)")

    # ── 3. Post-song voice clips (ADPCM, sync-run boundaries) ────────────────
    post_clips = get_postsong_clips(data, song_end)
    print(f"\n[post-song clips]  {len(post_clips)} clips found via sync-pattern")
    for j, (start, end) in enumerate(post_clips):
        size = end - start
        raw = out_dir / f"{name.lower()}_post_{j:03d}.bin"
        raw.write_bytes(data[start:end])
        result = find_best_shift(data[start:end], out_dir, f"{name.lower()}_post_{j:03d}")
        if result:
            shift, nibble_swap, wav, sfm, dur, clip_pct = result
            ns_tag = " nibble-swap" if nibble_swap else ""
            print(f"  post_{j:03d}  0x{start:07x} {size:7d}B  shift={shift:2d}B{ns_tag}  {wav.name}  {dur:.2f}s  clip={clip_pct:.0f}%  sfm={sfm:.3f}")
        else:
            print(f"  post_{j:03d}  0x{start:07x} {size:7d}B  (decode failed)")


def auto_detect_boundaries(data):
    """Fallback: detect song_start/song_end from byte std/near_zero."""
    window = 256
    samp = np.frombuffer(data, dtype=np.int8).astype(np.float32)
    stds = [samp[i:i+window].std() for i in range(0, len(samp)-window, window)]
    nzs  = [np.sum(np.abs(samp[i:i+window]) < 30)/window for i in range(0, len(samp)-window, window)]

    def is_song(std, nz): return std < 62 and nz > 0.33

    starts, ends = [], []
    in_song = False
    for k, (s, nz) in enumerate(zip(stds, nzs)):
        off = k * window
        if not in_song and is_song(s, nz) and off > 0x5650:
            starts.append(off)
            in_song = True
        elif in_song and not is_song(s, nz):
            ends.append(off)
            in_song = False
    if in_song:
        ends.append(len(data))

    song_start = starts[0] if starts else 0x35000
    # Largest contiguous run
    best = max(zip(starts, ends), key=lambda t: t[1]-t[0], default=(song_start, song_start+0xD0000))
    return {"song_start": best[0], "song_end": best[1]}


def main():
    ap = argparse.ArgumentParser(description="Extract audio from .bin toy sound files")
    ap.add_argument("files", nargs="+", help=".bin files to process")
    ap.add_argument("-o", "--outdir", default="extracted", help="Output directory (default: extracted/)")
    ap.add_argument("-r", "--rates", default="8000,11025,16000",
                    help="Sample rates to try for ADPCM (comma-separated)")
    args = ap.parse_args()

    rates = tuple(int(r) for r in args.rates.split(","))
    base_out = Path(args.outdir)

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"ERROR: {f} not found", file=sys.stderr)
            continue
        process_file(p, base_out / p.stem, rates)


if __name__ == "__main__":
    main()
