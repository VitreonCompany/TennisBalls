import subprocess
import imageio_ffmpeg
import os


# ============================================================
# PATHS
# ============================================================

INPUT_VIDEO = "videos/FedSlice.mp4"

OUTPUT_VIDEO = (
    "fake_validation_prototype/"
    "videos/"
    "federer_slice_web.mp4"
)


# ============================================================
# CLIP SETTINGS
# ============================================================

# Start at 31 seconds
START_TIME = "00:00:31"

# Finish at 39 seconds
DURATION = 8


# ============================================================
# CHECK SOURCE VIDEO
# ============================================================

if not os.path.exists(INPUT_VIDEO):
    print(f"Could not find: {INPUT_VIDEO}")
    raise SystemExit


# ============================================================
# GET FFMPEG
# ============================================================

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# CREATE WEB-COMPATIBLE CLIP
# ============================================================

command = [
    ffmpeg,
    "-y",

    "-ss",
    START_TIME,

    "-i",
    INPUT_VIDEO,

    "-t",
    str(DURATION),

    "-c:v",
    "libx264",

    "-preset",
    "fast",

    "-crf",
    "20",

    "-pix_fmt",
    "yuv420p",

    "-c:a",
    "aac",

    "-b:a",
    "128k",

    "-movflags",
    "+faststart",

    OUTPUT_VIDEO,
]


print("Creating Federer comparison clip...")

result = subprocess.run(command)


# ============================================================
# RESULT
# ============================================================

if result.returncode == 0:
    print()
    print("SUCCESS")
    print(f"Created: {OUTPUT_VIDEO}")
    print("Clip: 00:31 -> 00:39")
else:
    print()
    print("Something went wrong while creating the clip.")