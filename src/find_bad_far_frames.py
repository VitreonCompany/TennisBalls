import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "output/far_ground_tracking.csv"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV_PATH)


print()
print("BAD FAR-PLAYER FRAMES")
print("=" * 70)


# ============================================================
# FIND SUSPICIOUS DEPTH ESTIMATES
#
# For this 10-second test, we're simply looking for the
# measurements that went substantially behind the far baseline.
#
# This is diagnostic only — NOT a final rule.
# ============================================================

bad = df[
    (df["detected"] == True)
    &
    (df["court_y"] < -1.5)
].copy()


print(f"Suspicious frames: {len(bad)}")
print()


if len(bad) == 0:

    print("No suspicious frames found.")

else:

    print(
        bad[
            [
                "frame",
                "time_seconds",
                "confidence",
                "box_bottom_y",
                "ground_y",
                "offset",
                "court_x",
                "court_y",
                "image_evidence"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# GROUP CONSECUTIVE BAD FRAMES
# ============================================================

print()
print("BAD SECTIONS")
print("=" * 70)


if len(bad) > 0:

    bad_indices = bad.index.tolist()

    groups = []

    current_group = [
        bad_indices[0]
    ]


    for index in bad_indices[1:]:

        if (
            index
            ==
            current_group[-1] + 1
        ):

            current_group.append(
                index
            )

        else:

            groups.append(
                current_group
            )

            current_group = [
                index
            ]


    groups.append(
        current_group
    )


    for number, group in enumerate(
        groups,
        start=1
    ):

        first = df.loc[
            group[0]
        ]

        last = df.loc[
            group[-1]
        ]

        minimum_y = df.loc[
            group,
            "court_y"
        ].min()


        print(
            f"Section {number}: "
            f"{first['time_seconds']:.2f}s "
            f"-> "
            f"{last['time_seconds']:.2f}s "
            f"| {len(group)} frames "
            f"| minimum={minimum_y:.2f}m"
        )


# ============================================================
# WORST 20
# ============================================================

print()
print("WORST 20 MEASUREMENTS")
print("=" * 70)


worst = (
    df[
        df["detected"] == True
    ]
    .sort_values(
        "court_y"
    )
    .head(20)
)


print(
    worst[
        [
            "frame",
            "time_seconds",
            "confidence",
            "box_bottom_y",
            "ground_y",
            "offset",
            "court_x",
            "court_y",
            "image_evidence"
        ]
    ].to_string(
        index=False
    )
)