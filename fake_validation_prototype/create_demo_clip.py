import subprocess
import imageio_ffmpeg
import os


# ============================================================
# PATHS
# ============================================================

INPUT_VIDEO = "videos/test_match.mp4"

OUTPUT_VIDEO = (
    "fake_validation_prototype/"
    "videos/demo_point_web.mp4"
)


# ============================================================
# CLIP TIMES
#
# 10:03 -> 10:14
# ============================================================

START_TIME = "00:10:03"
DURATION = 11


# ============================================================
# GET BUNDLED FFMPEG
# ============================================================

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

print()
print("=" * 60)
print("CREATING BROWSER-COMPATIBLE DEMO CLIP")
print("=" * 60)

print()
print(f"FFmpeg: {ffmpeg}")
print(f"Input:  {INPUT_VIDEO}")
print(f"Output: {OUTPUT_VIDEO}")
print()


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_VIDEO),
    exist_ok=True
)


# ============================================================
# FFMPEG COMMAND
#
# H.264 + AAC is much more browser friendly than OpenCV's
# mp4v output.
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


print("Encoding 11-second clip...")
print()


result = subprocess.run(
    command
)


if result.returncode != 0:

    raise RuntimeError(
        "FFmpeg failed to create the demo clip."
    )


print()
print("=" * 60)
print("DONE")
print("=" * 60)

print()
print(
    f"Saved: {OUTPUT_VIDEO}"
)