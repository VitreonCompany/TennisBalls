import numpy as np


# ============================================================
# AUTOMATIC DETECTIONS FROM OUR CV PIPELINE
#
# These are NOT manual points.
# ============================================================

LEFT_DOUBLES = np.array(
    [685.2, 786.6],
    dtype=np.float64
)

LEFT_SINGLES = np.array(
    [942.7, 736.6],
    dtype=np.float64
)

RIGHT_DOUBLES = np.array(
    [1698.9, 589.9],
    dtype=np.float64
)


# ============================================================
# REAL TENNIS DIMENSIONS
# ============================================================

DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23

ALLEY_WIDTH = (
    DOUBLES_WIDTH - SINGLES_WIDTH
) / 2.0


print()
print("TENNIS DIMENSIONS")
print("=" * 60)

print(
    f"Doubles width: {DOUBLES_WIDTH:.2f} m"
)

print(
    f"Singles width: {SINGLES_WIDTH:.2f} m"
)

print(
    f"Each alley: {ALLEY_WIDTH:.2f} m"
)


# ============================================================
# PARAMETERISE THE IMAGE BASELINE
#
# We represent any point on the detected near baseline using
# scalar coordinate t:
#
# left doubles  -> t = 0
# right doubles -> t = 1
#
# Image point = LEFT + t * (RIGHT - LEFT)
# ============================================================

baseline_vector = (
    RIGHT_DOUBLES - LEFT_DOUBLES
)


def image_t(point):

    return np.dot(
        point - LEFT_DOUBLES,
        baseline_vector
    ) / np.dot(
        baseline_vector,
        baseline_vector
    )


t_left_doubles = 0.0

t_left_singles = image_t(
    LEFT_SINGLES
)

t_right_doubles = 1.0


print()
print("IMAGE BASELINE COORDINATES")
print("=" * 60)

print(
    f"Left doubles t  = "
    f"{t_left_doubles:.6f}"
)

print(
    f"Left singles t  = "
    f"{t_left_singles:.6f}"
)

print(
    f"Right doubles t = "
    f"{t_right_doubles:.6f}"
)


# ============================================================
# 1D PROJECTIVE TRANSFORM
#
# Real baseline coordinate X maps to image coordinate t:
#
#       aX + b
# t = -----------
#       cX + 1
#
# We know three correspondences:
#
# X = 0.00       -> left doubles
# X = 1.37       -> left singles
# X = 10.97      -> right doubles
#
# Three correspondences determine this 1D projective mapping.
#
# Then evaluate it at:
#
# X = 9.60
#
# which is the right singles sideline.
# ============================================================

X1 = 0.0
T1 = t_left_doubles

X2 = ALLEY_WIDTH
T2 = t_left_singles

X3 = DOUBLES_WIDTH
T3 = t_right_doubles


# Because X1 = 0 and T1 = 0:
#
# b = 0
#
# Therefore:
#
# t = aX / (cX + 1)
#
# Solve using X2/T2 and X3/T3.

A = np.array(
    [
        [
            X2,
            -T2 * X2
        ],
        [
            X3,
            -T3 * X3
        ]
    ],
    dtype=np.float64
)

B = np.array(
    [
        T2,
        T3
    ],
    dtype=np.float64
)


a, c = np.linalg.solve(
    A,
    B
)


print()
print("PROJECTIVE MODEL")
print("=" * 60)

print(f"a = {a:.8f}")
print(f"c = {c:.8f}")


# ============================================================
# RIGHT SINGLES REAL POSITION
# ============================================================

X_RIGHT_SINGLES = (
    DOUBLES_WIDTH - ALLEY_WIDTH
)


def project_real_x(X):

    return (
        a * X
    ) / (
        c * X + 1.0
    )


t_right_singles = project_real_x(
    X_RIGHT_SINGLES
)


# ============================================================
# CONVERT t BACK TO IMAGE PIXELS
# ============================================================

P8 = (
    LEFT_DOUBLES
    +
    t_right_singles
    * baseline_vector
)


print()
print("INFERRED RIGHT SINGLES CORNER")
print("=" * 60)

print(
    f"Real X = "
    f"{X_RIGHT_SINGLES:.3f} m"
)

print(
    f"Image t = "
    f"{t_right_singles:.6f}"
)

print(
    f"P8 = "
    f"({P8[0]:.1f}, {P8[1]:.1f})"
)


# ============================================================
# HIDDEN MANUAL BENCHMARK
#
# NOT USED IN CALCULATION.
# ============================================================

manual = np.load(
    "data/court_points_8.npy"
).astype(np.float64)

manual_p8 = manual[7]


error = np.linalg.norm(
    P8 - manual_p8
)


print()
print("HIDDEN MANUAL BENCHMARK")
print("=" * 60)

print(
    f"AUTO P8 = "
    f"({P8[0]:.1f}, {P8[1]:.1f})"
)

print(
    f"MANUAL P8 = "
    f"({manual_p8[0]:.1f}, "
    f"{manual_p8[1]:.1f})"
)

print(
    f"P8 ERROR = "
    f"{error:.1f}px"
)