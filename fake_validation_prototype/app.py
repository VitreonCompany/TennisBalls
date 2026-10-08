import os
import json
import time
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from match_data import MATCH_DATA


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tennis AI",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# OPENAI
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY was not found.")
    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# COLOURS
# ============================================================

GREEN = "#063B26"
DEEP_GREEN = "#032C1C"
PURPLE = "#5A1A78"
TENNIS_LIME = "#D9E86C"
WHITE = "#FFFFFF"
TEXT = "#10281C"


# ============================================================
# MAIN CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background: #FFFFFF !important;
    }}

    .block-container {{
        max-width: 1450px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 5rem !important;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header[data-testid="stHeader"] {{
        background: #FFFFFF !important;
    }}

    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp li {{
        color: {TEXT};
    }}

    .stApp h1 {{
        color: {GREEN} !important;
        font-size: 44px !important;
        font-weight: 900 !important;
        letter-spacing: -1.5px !important;
    }}

    .stApp h2 {{
        color: {GREEN} !important;
        font-size: 30px !important;
        font-weight: 900 !important;
        letter-spacing: -0.8px !important;
    }}

    .stApp h3 {{
        color: {GREEN} !important;
        font-size: 20px !important;
        font-weight: 850 !important;
    }}

    strong {{
        color: {GREEN} !important;
        font-weight: 850 !important;
    }}

    [data-testid="stCaptionContainer"] p {{
        color: {PURPLE} !important;
        font-size: 11px !important;
        font-weight: 850 !important;
        letter-spacing: 0.9px !important;
    }}

    video {{
        border-radius: 3px !important;
        border-bottom: 4px solid {PURPLE} !important;
        box-shadow: 0 7px 22px rgba(0, 35, 18, 0.12) !important;
    }}

    [data-testid="stChatMessage"] {{
        background: #FFFFFF !important;
        border: 1px solid #D8DDD9 !important;
        border-radius: 3px !important;
        padding: 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 3px 12px rgba(0, 30, 15, 0.05) !important;
    }}

    [data-testid="stChatMessage"] p {{
        color: {TEXT} !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
    }}

    [data-testid="stChatInput"] {{
        background: #FFFFFF !important;
        border: 2px solid {GREEN} !important;
        border-radius: 3px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        background: #FFFFFF !important;
        color: {TEXT} !important;
        font-size: 15px !important;
    }}

    [data-testid="stMetric"] {{
        background: #FFFFFF !important;
        border: 1px solid #D8DDD9 !important;
        border-top: 4px solid {GREEN} !important;
        border-radius: 2px !important;
        padding: 17px !important;
    }}

    [data-testid="stMetricLabel"] p {{
        color: {PURPLE} !important;
        font-size: 11px !important;
        font-weight: 850 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.7px !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {GREEN} !important;
        font-size: 27px !important;
        font-weight: 900 !important;
    }}

    [data-testid="stAlert"] {{
        background: #EDF4F0 !important;
        border: 1px solid #D4DFD8 !important;
        border-left: 6px solid {PURPLE} !important;
        border-radius: 2px !important;
    }}

    [data-testid="stAlert"] p {{
        color: {TEXT} !important;
    }}

    [data-testid="stStatusWidget"] {{
        background: {GREEN} !important;
        border: none !important;
        border-left: 6px solid {PURPLE} !important;
        border-radius: 2px !important;
    }}

    [data-testid="stStatusWidget"] p,
    [data-testid="stStatusWidget"] span {{
        color: #FFFFFF !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid #D8DDD9 !important;
        border-radius: 3px !important;
        box-shadow: 0 4px 15px rgba(0, 35, 18, 0.04);
    }}

    hr {{
        border: none !important;
        border-top: 1px solid #D8DDD9 !important;
        margin-top: 2.5rem !important;
        margin-bottom: 2.5rem !important;
    }}

    ::-webkit-scrollbar {{
        width: 9px;
    }}

    ::-webkit-scrollbar-track {{
        background: #F2F3F0;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {GREEN};
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {PURPLE};
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

player = MATCH_DATA["player"]
match = MATCH_DATA["match"]


DEMO_ANALYSIS = {
    "current_point": {
        "time": "10:03–10:14",
        "score": "5-3, 15-30",
        "result": "Lost",
        "shot": "Backhand slice",
        "shot_choice":
            "The backhand slice was the correct tactical choice.",
        "problem":
            "The depth was reasonably good. The issue was the "
            "quality of the slice. It sat up and did not cut "
            "through the court enough, giving the opponent a "
            "comfortable attacking ball down the line.",
        "better_execution":
            "Keep the slice lower and make it penetrate through "
            "the court more effectively.",
        "tactical_goal":
            "Neutralise the opponent, buy recovery time and "
            "receive a more manageable next ball."
    },

    "professional_reference": {
        "player": "Roger Federer",
        "scenario":
            "Comparable backhand slice from a similar rally situation.",
        "lesson":
            "Federer's slice stays lower and penetrates through "
            "the court more effectively.",
        "key_difference":
            "Trajectory, penetration and effectiveness."
    },

    "match_pattern": {
        "similar_instances": 6,
        "won": 1,
        "lost": 5,
        "success_rate": "17%"
    },

    "recommended_drill": {
        "name": "Penetrating Slice Drill",
        "duration": "10 minutes",
        "focus": "Slice quality",
        "goal": "Neutralise the opponent"
    }
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are the AI Match Coach inside an advanced tennis analysis
application.

You are speaking directly to the player whose match is displayed.

The system has analysed the point, searched the match for
similar situations and identified a relevant professional
reference example.

Use ONLY the information below.

CURRENT ANALYSIS:

{json.dumps(DEMO_ANALYSIS, indent=2)}

PLAYER DATA:

{json.dumps(MATCH_DATA, indent=2)}

COACHING STYLE:

Speak like an experienced tennis coach.

Be specific, tactical and actionable.

WHEN ASKED WHAT WENT WRONG:

Use these sections:

**What went wrong**

The backhand slice was the correct tactical choice.

Do NOT say depth was the main problem.

The depth was reasonably good.

The issue was the QUALITY of the slice.

It sat up and did not penetrate through the court enough.

That gave the opponent time to step into the ball,
take a comfortable full swing and attack down the line.

**What to do next time**

Still use the slice.

Focus on:

- lower trajectory
- more penetration through the court
- preventing the ball sitting up
- making the opponent contact a more difficult ball
- recovering immediately afterwards

The objective is not to hit a winner.

The objective is to turn a pressured situation into
a neutral rally.

If the opponent cannot comfortably attack the next ball,
the slice has done its job.

**Professional comparison**

Tell the player that a comparable Roger Federer example
has been identified and is shown alongside the analysis.

Explain what they should watch for:

- lower trajectory
- penetration through the court
- ball staying lower
- opponent unable to comfortably attack
- Federer recovering after the shot

Do NOT invent professional statistics.

**Pattern in your match**

Mention naturally:

"I identified 6 other comparable situations in this match."

The player won 1 of those 6 situations.

Do not make this the main focus.

Use it to show this may be a recurring tactical pattern.

**What I'd work on**

Recommend the Penetrating Slice Drill.

The training question is:

"Can my opponent comfortably attack my slice?"

If yes, it was not effective enough.

If they have to send back a neutral ball,
the slice has done its job.

IMPORTANT:

Keep the full analysis around 180-260 words.

Use bold headings.

Prioritise actionable advice.

Do not say you personally watched the video.

Do not diagnose medical conditions.
"""


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


# ============================================================
# TOP HEADER
# ============================================================

def top_header():

    components.html(
        f"""
<!DOCTYPE html>

<html>

<head>

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;
    overflow: hidden;
    font-family: Arial, Helvetica, sans-serif;
}}

.header {{
    background: {GREEN};
    border: 1px solid {DEEP_GREEN};
    border-bottom: 5px solid {PURPLE};
    border-radius: 4px;

    height: 155px;

    padding: 0 32px;

    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.brand-section {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 15px;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

.icon {{
    font-size: 40px;
    line-height: 1;
}}

.brand-name {{
    color: white;
    font-size: 38px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -1.3px;
}}

.subtitle {{
    color: {TENNIS_LIME};
    font-size: 11px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: 1.5px;
}}

.nav {{
    color: white;
    font-size: 13px;
    line-height: 1.3;
    font-weight: 900;
    letter-spacing: .2px;

    display: flex;
    align-items: center;
    height: 100%;
}}

@media (max-width: 750px) {{

    .header {{
        height: 190px;
        padding: 25px;
        display: block;
    }}

    .brand-section {{
        margin-top: 15px;
    }}

    .nav {{
        height: auto;
        margin-top: 25px;
    }}

}}

</style>

</head>


<body>

<div class="header">

    <div class="brand-section">

        <div class="brand">

            <div class="icon">
                🎾
            </div>

            <div class="brand-name">
                Tennis AI
            </div>

        </div>

        <div class="subtitle">
            AI MATCH INTELLIGENCE
        </div>

    </div>

    <div class="nav">
        MATCH ANALYSIS · DEVELOPMENT · PLAYER
    </div>

</div>

</body>

</html>
        """,
        height=175,
        scrolling=False,
    )


# ============================================================
# GREEN SECTION BANNERS
# ============================================================

def green_banner(label, title, text=""):

    components.html(
        f"""
<!DOCTYPE html>

<html>

<head>

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;
    overflow: hidden;
    font-family: Arial, Helvetica, sans-serif;
}}

.banner {{
    width: 100%;
    height: 250px;

    background: {GREEN};

    border-bottom: 5px solid {PURPLE};

    display: flex;
    align-items: center;

    padding: 0 50px;
}}

.content {{
    width: 100%;
    max-width: 950px;
}}

.label {{
    color: {TENNIS_LIME};

    font-size: 12px;
    font-weight: 900;

    letter-spacing: 2px;

    text-transform: uppercase;

    margin-bottom: 14px;
}}

.title {{
    color: white;

    font-size: 38px;
    font-weight: 900;

    line-height: 1.08;

    letter-spacing: -0.7px;

    margin-bottom: 16px;
}}

.description {{
    color: #E8F0EB;

    font-size: 16px;

    line-height: 1.6;

    max-width: 850px;
}}

@media (max-width: 750px) {{

    .banner {{
        height: 260px;
        padding: 0 28px;
    }}

    .title {{
        font-size: 31px;
    }}

    .description {{
        font-size: 15px;
    }}

}}

</style>

</head>


<body>

<div class="banner">

    <div class="content">

        <div class="label">
            {label}
        </div>

        <div class="title">
            {title}
        </div>

        <div class="description">
            {text}
        </div>

    </div>

</div>

</body>

</html>
        """,
        height=270,
        scrolling=False,
    )


# ============================================================
# THREE TENNIS COURTS
# ============================================================

def training_court_diagrams():

    components.html(
        f"""
<!DOCTYPE html>

<html>

<head>

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;

    background: white;

    font-family: Arial, Helvetica, sans-serif;

    color: {TEXT};
}}


/* ==========================================================
   DESKTOP WRAPPER
========================================================== */

.wrapper {{
    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 22px;

    width: 100%;
}}


/* ==========================================================
   CARD
========================================================== */

.card {{
    min-width: 0;

    border:
        1px solid #D8DDD9;

    background:
        #FFFFFF;

    overflow: hidden;
}}


.card-top {{
    padding:
        20px
        20px
        14px
        20px;
}}


.number {{
    color: {PURPLE};

    font-size: 11px;

    letter-spacing: 1.5px;

    font-weight: 900;

    margin-bottom: 8px;
}}


.title {{
    color: {GREEN};

    font-size: 20px;

    font-weight: 900;

    margin-bottom: 6px;
}}


.subtitle {{
    color: #6B786F;

    font-size: 13px;

    line-height: 1.45;

    min-height: 38px;
}}


/* ==========================================================
   COURT
========================================================== */

.court-area {{
    background: {GREEN};

    padding: 22px;

    border-top:
        4px solid {PURPLE};
}}


.court {{
    position: relative;

    width: 230px;

    height: 400px;

    margin: 0 auto;

    border:
        3px solid white;
}}


.single-left {{
    position: absolute;

    left: 22px;

    top: 0;

    width: 2px;

    height: 100%;

    background: white;
}}


.single-right {{
    position: absolute;

    right: 22px;

    top: 0;

    width: 2px;

    height: 100%;

    background: white;
}}


.service-top {{
    position: absolute;

    left: 22px;

    right: 22px;

    top: 112px;

    height: 2px;

    background: white;
}}


.service-bottom {{
    position: absolute;

    left: 22px;

    right: 22px;

    bottom: 112px;

    height: 2px;

    background: white;
}}


.center-top {{
    position: absolute;

    left: 50%;

    top: 112px;

    width: 2px;

    height: 88px;

    background: white;
}}


.center-bottom {{
    position: absolute;

    left: 50%;

    bottom: 112px;

    width: 2px;

    height: 88px;

    background: white;
}}


.net {{
    position: absolute;

    left: -8px;

    right: -8px;

    top: 50%;

    height: 4px;

    background: {TENNIS_LIME};
}}


.mark-top {{
    position: absolute;

    width: 2px;

    height: 10px;

    background: white;

    top: 0;

    left: 50%;
}}


.mark-bottom {{
    position: absolute;

    width: 2px;

    height: 10px;

    background: white;

    bottom: 0;

    left: 50%;
}}


/* ==========================================================
   PLAYERS
========================================================== */

.player {{
    position: absolute;

    width: 25px;

    height: 25px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 10px;

    font-weight: 900;

    z-index: 10;
}}


.you {{
    background: white;

    color: {GREEN};

    border:
        3px solid {PURPLE};
}}


.opponent {{
    background: {TENNIS_LIME};

    color: {GREEN};

    border:
        3px solid white;
}}


/* ==========================================================
   BALL
========================================================== */

.ball-dot {{
    position: absolute;

    width: 13px;

    height: 13px;

    border-radius: 50%;

    background: {TENNIS_LIME};

    border:
        2px solid white;

    z-index: 12;
}}


/* ==========================================================
   SHOT PATHS
========================================================== */

.path {{
    position: absolute;

    height: 4px;

    transform-origin:
        left center;

    z-index: 5;
}}


.path::after {{
    content: "";

    position: absolute;

    right: -1px;

    top: -5px;

    border-left:
        10px solid currentColor;

    border-top:
        6px solid transparent;

    border-bottom:
        6px solid transparent;
}}


.user-path {{
    background:
        #D3A5E6;

    color:
        #D3A5E6;
}}


.attack-path {{
    background:
        {TENNIS_LIME};

    color:
        {TENNIS_LIME};
}}


.good-path {{
    background:
        white;

    color:
        white;
}}


/* ==========================================================
   TARGET
========================================================== */

.target-zone {{
    position: absolute;

    background:
        repeating-linear-gradient(
            -45deg,
            rgba(217,232,108,.95),
            rgba(217,232,108,.95) 8px,
            rgba(217,232,108,.55) 8px,
            rgba(217,232,108,.55) 16px
        );

    border:
        2px solid white;

    z-index: 2;
}}


/* ==========================================================
   COURT LABELS
========================================================== */

.court-label {{
    position: absolute;

    background:
        rgba(3,44,28,.94);

    color:
        white;

    padding:
        5px 7px;

    font-size:
        10px;

    font-weight:
        800;

    white-space:
        nowrap;

    z-index: 20;
}}


.purple-label {{
    border-left:
        3px solid #D3A5E6;
}}


.yellow-label {{
    border-left:
        3px solid {TENNIS_LIME};
}}


/* ==========================================================
   EXPLANATION
========================================================== */

.explanation {{
    padding:
        18px
        20px
        22px
        20px;
}}


.explanation strong {{
    color: {GREEN};
}}


.explanation p {{
    font-size: 13px;

    line-height: 1.55;

    margin: 0;
}}


.result {{
    margin-top: 15px;

    padding:
        10px
        12px;

    font-size: 12px;

    line-height: 1.45;

    font-weight: 700;
}}


.bad {{
    background:
        #F5EAF7;

    border-left:
        4px solid {PURPLE};
}}


.good {{
    background:
        #EEF4E4;

    border-left:
        4px solid #8FAE3E;
}}


/* ==========================================================
   SWIPE HINT
========================================================== */

.swipe-hint {{
    display: none;
}}


/* ==========================================================
   MOBILE
========================================================== */

@media (max-width: 900px) {{

    html,
    body {{
        overflow-x: hidden;
    }}


    .swipe-hint {{
        display: flex;

        align-items: center;

        justify-content: space-between;

        margin:
            0
            14px
            12px
            14px;

        color:
            {PURPLE};

        font-size:
            11px;

        font-weight:
            900;

        letter-spacing:
            1px;
    }}


    .swipe-arrow {{
        color:
            {GREEN};

        font-size:
            16px;
    }}


    .wrapper {{
        display: flex;

        grid-template-columns: none;

        width: 100%;

        gap: 14px;

        overflow-x: auto;

        overflow-y: hidden;

        scroll-snap-type:
            x mandatory;

        scroll-padding:
            14px;

        -webkit-overflow-scrolling:
            touch;

        overscroll-behavior-x:
            contain;

        padding:
            0
            14px
            22px
            14px;

        scrollbar-width:
            none;
    }}


    .wrapper::-webkit-scrollbar {{
        display: none;
    }}


    .card {{
        flex:
            0 0
            calc(100% - 28px);

        width:
            calc(100% - 28px);

        min-width:
            calc(100% - 28px);

        scroll-snap-align:
            center;

        scroll-snap-stop:
            always;
    }}


    .card-top {{
        padding:
            18px
            18px
            14px
            18px;
    }}


    .court-area {{
        padding:
            20px
            8px;
    }}


    .court {{
        width: 230px;

        max-width: 100%;

        height: 400px;
    }}


    .explanation {{
        padding:
            18px;
    }}

}}

</style>

</head>


<body>


<div class="swipe-hint">

    <span class="swipe-arrow">
        ←
    </span>

    <span>
        SWIPE THROUGH THE ANALYSIS
    </span>

    <span class="swipe-arrow">
        →
    </span>

</div>


<div class="wrapper">


<!-- ========================================================
     COURT 1
======================================================== -->

<div class="card">

    <div class="card-top">

        <div class="number">
            01 · YOUR POINT
        </div>

        <div class="title">
            The slice sits up
        </div>

        <div class="subtitle">
            Good enough depth, but the ball does not penetrate
            through the court.
        </div>

    </div>


    <div class="court-area">

        <div class="court">

            <div class="single-left"></div>
            <div class="single-right"></div>

            <div class="service-top"></div>
            <div class="service-bottom"></div>

            <div class="center-top"></div>
            <div class="center-bottom"></div>

            <div class="net"></div>

            <div class="mark-top"></div>
            <div class="mark-bottom"></div>


            <div
                class="player opponent"
                style="
                    left:102px;
                    top:58px;
                "
            >
                O
            </div>


            <div
                class="court-label yellow-label"
                style="
                    left:135px;
                    top:55px;
                "
            >
                Steps forward
            </div>


            <div
                class="player you"
                style="
                    left:72px;
                    bottom:22px;
                "
            >
                YOU
            </div>


            <div
                class="path user-path"
                style="
                    left:84px;
                    bottom:54px;
                    width:190px;
                    transform:rotate(-67deg);
                "
            ></div>


            <div
                class="ball-dot"
                style="
                    left:112px;
                    top:108px;
                "
            ></div>


            <div
                class="court-label purple-label"
                style="
                    left:15px;
                    top:145px;
                "
            >
                Slice sits up
            </div>


            <div
                class="path attack-path"
                style="
                    left:111px;
                    top:82px;
                    width:220px;
                    transform:rotate(67deg);
                "
            ></div>


            <div
                class="court-label yellow-label"
                style="
                    right:8px;
                    top:235px;
                "
            >
                Attacks DTL
            </div>

        </div>

    </div>


    <div class="explanation">

        <p>
            Your opponent is able to
            <strong>step inside the court</strong>
            and take a comfortable attacking swing.
        </p>

        <div class="result bad">
            PROBLEM → The next ball is attacking.
        </div>

    </div>

</div>


<!-- ========================================================
     COURT 2
======================================================== -->

<div class="card">

    <div class="card-top">

        <div class="number">
            02 · BETTER EXECUTION
        </div>

        <div class="title">
            Neutralise the rally
        </div>

        <div class="subtitle">
            Same tactical choice — but a lower and more
            penetrating slice.
        </div>

    </div>


    <div class="court-area">

        <div class="court">

            <div class="single-left"></div>
            <div class="single-right"></div>

            <div class="service-top"></div>
            <div class="service-bottom"></div>

            <div class="center-top"></div>
            <div class="center-bottom"></div>

            <div class="net"></div>

            <div class="mark-top"></div>
            <div class="mark-bottom"></div>


            <div
                class="player opponent"
                style="
                    left:102px;
                    top:18px;
                "
            >
                O
            </div>


            <div
                class="court-label yellow-label"
                style="
                    left:135px;
                    top:17px;
                "
            >
                Kept back
            </div>


            <div
                class="player you"
                style="
                    left:72px;
                    bottom:22px;
                "
            >
                YOU
            </div>


            <div
                class="path good-path"
                style="
                    left:84px;
                    bottom:54px;
                    width:320px;
                    transform:rotate(-74deg);
                "
            ></div>


            <div
                class="ball-dot"
                style="
                    left:110px;
                    top:63px;
                "
            ></div>


            <div
                class="court-label purple-label"
                style="
                    left:8px;
                    top:140px;
                "
            >
                Lower + penetrating
            </div>


            <div
                class="path attack-path"
                style="
                    left:110px;
                    top:44px;
                    width:180px;
                    transform:rotate(72deg);
                    opacity:.75;
                "
            ></div>


            <div
                class="court-label yellow-label"
                style="
                    right:8px;
                    top:205px;
                "
            >
                Neutral reply
            </div>

        </div>

    </div>


    <div class="explanation">

        <p>
            The opponent has to contact the ball from a
            <strong>less aggressive court position</strong>.
        </p>

        <div class="result good">
            GOAL → Pressure becomes neutral.
        </div>

    </div>

</div>


<!-- ========================================================
     COURT 3
======================================================== -->

<div class="card">

    <div class="card-top">

        <div class="number">
            03 · YOUR DRILL
        </div>

        <div class="title">
            Train the solution
        </div>

        <div class="subtitle">
            Practise the same slice and judge it by the
            opponent's next ball.
        </div>

    </div>


    <div class="court-area">

        <div class="court">

            <div class="single-left"></div>
            <div class="single-right"></div>

            <div class="service-top"></div>
            <div class="service-bottom"></div>

            <div class="center-top"></div>
            <div class="center-bottom"></div>

            <div class="net"></div>

            <div class="mark-top"></div>
            <div class="mark-bottom"></div>


            <div
                class="target-zone"
                style="
                    left:24px;
                    right:24px;
                    top:25px;
                    height:70px;
                "
            ></div>


            <div
                class="court-label yellow-label"
                style="
                    left:53px;
                    top:100px;
                "
            >
                Penetration zone
            </div>


            <div
                class="player opponent"
                style="
                    left:102px;
                    top:10px;
                "
            >
                O
            </div>


            <div
                class="player you"
                style="
                    left:72px;
                    bottom:22px;
                "
            >
                YOU
            </div>


            <div
                class="path good-path"
                style="
                    left:84px;
                    bottom:54px;
                    width:320px;
                    transform:rotate(-74deg);
                "
            ></div>


            <div
                class="ball-dot"
                style="
                    left:108px;
                    top:66px;
                "
            ></div>


            <div
                class="court-label purple-label"
                style="
                    left:8px;
                    top:145px;
                "
            >
                Drive through it
            </div>

        </div>

    </div>


    <div class="explanation">

        <p>
            Don't judge the repetition only by where the ball lands.
            Judge <strong>what your opponent can do next</strong>.
        </p>

        <div class="result good">
            SUCCESS → They cannot comfortably attack.
        </div>

    </div>

</div>


</div>

</body>

</html>
        """,
        height=800,
        scrolling=False,
    )


# ============================================================
# TOP HEADER
# ============================================================

top_header()


# ============================================================
# HERO TEXT
# ============================================================

st.write("")

st.caption(
    "YOUR MATCH • UNDERSTOOD"
)

st.header(
    "Understand the point. Improve the next one."
)

st.write(
    "AI match analysis that explains what happened, "
    "shows how higher-level players solve the same situation "
    "and turns the insight into something you can train."
)

st.divider()


# ============================================================
# MAIN ANALYSIS
# ============================================================

video_col, chat_col = st.columns(
    [1.05, 1],
    gap="large"
)


# ============================================================
# LEFT — YOUR POINT
# ============================================================

with video_col:

    st.caption(
        "POINT 47 • MATCH ANALYSIS"
    )

    st.header(
        "Your Point"
    )


    user_video_path = (
        "fake_validation_prototype/"
        "videos/"
        "demo_point_web.mp4"
    )


    if os.path.exists(user_video_path):

        with open(
            user_video_path,
            "rb"
        ) as video_file:

            st.video(
                video_file.read(),
                format="video/mp4"
            )

    else:

        st.warning(
            "demo_point_web.mp4 was not found."
        )


    # ========================================================
    # PROFESSIONAL COMPARISON
    # ========================================================

    if st.session_state.analysis_complete:

        st.write("")

        professional = st.container(
            border=True
        )

        with professional:

            st.caption(
                "PROFESSIONAL COMPARISON FOUND"
            )

            st.header(
                "Roger Federer"
            )

            st.write(
                "**Comparable backhand slice situation**"
            )

            st.caption(
                "PROTOTYPE PROFESSIONAL REFERENCE"
            )


            federer_video_path = (
                "fake_validation_prototype/"
                "videos/"
                "federer_slice_web.mp4"
            )


            if os.path.exists(federer_video_path):

                with open(
                    federer_video_path,
                    "rb"
                ) as federer_video:

                    st.video(
                        federer_video.read(),
                        format="video/mp4"
                    )

            else:

                st.warning(
                    "federer_slice_web.mp4 was not found."
                )


            st.subheader(
                "What to watch"
            )


            your_shot, pro_shot = st.columns(
                2
            )


            with your_shot:

                st.caption(
                    "YOUR SLICE"
                )

                st.write(
                    "✓ Reasonable depth"
                )

                st.write(
                    "⚠️ Sits up"
                )

                st.write(
                    "⚠️ Less penetration"
                )

                st.write(
                    "❌ Attackable next ball"
                )


            with pro_shot:

                st.caption(
                    "FEDERER"
                )

                st.write(
                    "✓ Good depth"
                )

                st.write(
                    "✓ Lower trajectory"
                )

                st.write(
                    "✓ Cuts through court"
                )

                st.write(
                    "✓ Neutralising next ball"
                )


            st.info(
                "🎯 **Key difference:** It isn't simply depth. "
                "Federer's slice stays lower and penetrates "
                "through the court more effectively, making "
                "the next ball harder to attack."
            )


    # ========================================================
    # MATCH CONTEXT
    # ========================================================

    st.divider()

    st.caption(
        "MATCH CONTEXT"
    )

    st.subheader(
        "Match Situation"
    )


    score_col, result_col = st.columns(
        2
    )


    with score_col:

        st.metric(
            "Score",
            "5–3, 15–30"
        )


    with result_col:

        st.metric(
            "Point Result",
            "Lost"
        )


    st.write(
        f"**{player['name']} vs "
        f"{match['opponent']}**"
    )


    st.caption(
        f"{match['surface']} • "
        f"Point 47 • "
        f"10:03–10:14"
    )


# ============================================================
# RIGHT — AI COACH
# ============================================================

with chat_col:

    st.caption(
        "AI MATCH COACH"
    )

    st.header(
        "Match Intelligence"
    )

    st.caption(
        "ASK ANYTHING ABOUT THIS POINT"
    )


    if len(st.session_state.messages) == 0:

        st.info(
            "🎾 Try asking: **What went wrong here "
            "and what should I do differently?**"
        )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # INPUT
    # ========================================================

    question = st.chat_input(
        "Ask your AI coach..."
    )


    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        api_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]


        api_messages.extend(
            st.session_state.messages
        )


        # ====================================================
        # ANALYSIS
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            status = st.status(
                "Analysing your point...",
                expanded=True
            )


            with status:

                st.write(
                    "🎾 Understanding the tactical situation..."
                )

                time.sleep(0.6)


                st.write(
                    "🔎 Searching professional reference examples..."
                )

                time.sleep(0.8)


                st.write(
                    "📊 Searching this match for similar situations..."
                )

                time.sleep(0.6)


                st.write(
                    "🎯 Building personalised training recommendation..."
                )


                try:

                    response = (
                        client
                        .chat
                        .completions
                        .create(
                            model="gpt-5.6",
                            messages=api_messages
                        )
                    )


                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )


                    status.update(
                        label=(
                            "Analysis complete • "
                            "Professional comparison found"
                        ),
                        state="complete",
                        expanded=False
                    )


                except Exception as error:

                    answer = (
                        "I couldn't connect to the AI service.\n\n"
                        f"{error}"
                    )


                    status.update(
                        label="Analysis failed",
                        state="error",
                        expanded=False
                    )


            st.markdown(
                answer
            )


        # ====================================================
        # SAVE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        st.session_state.analysis_complete = True

        st.rerun()


# ============================================================
# AFTER ANALYSIS
# ============================================================

if st.session_state.analysis_complete:


    # ========================================================
    # ANALYSIS TO ACTION
    # ========================================================

    st.write("")
    st.write("")

    green_banner(
        "FROM ANALYSIS TO ACTION",
        "Now train the solution.",
        (
            "Your match showed us the problem. "
            "The professional example showed us what better "
            "execution looks like. Now turn that insight "
            "into something you can practise."
        )
    )


    # ========================================================
    # TRAINING PLAN
    # ========================================================

    st.write("")
    st.write("")

    st.caption(
        "PERSONALISED DEVELOPMENT"
    )

    st.header(
        "Your Training Plan"
    )


    training_left, training_right = st.columns(
        [0.75, 1.55],
        gap="large"
    )


    with training_left:

        st.subheader(
            "Penetrating Slice Drill"
        )


        st.metric(
            "Duration",
            "10 min"
        )


        st.write(
            "**Focus:** Slice quality"
        )


        st.write(
            "**Goal:** Neutralise the opponent"
        )


    with training_right:

        st.subheader(
            "How To Train It"
        )


        st.write(
            "Start each repetition with a ball to your "
            "backhand that encourages you to use the slice."
        )


        st.write(
            "Focus on producing a **low trajectory** and "
            "getting the ball to **cut through the court "
            "rather than sitting up**."
        )


        st.write(
            "Your partner should actively try to attack "
            "the next ball."
        )


        st.info(
            "🎯 **Success:** Your opponent cannot comfortably "
            "take a full attacking swing and instead has to "
            "play a neutral ball."
        )


        st.write(
            "Recover immediately after every slice and "
            "prepare for the next shot."
        )


    # ========================================================
    # VISUAL COACHING
    # ========================================================

    st.write("")
    st.write("")

    green_banner(
        "VISUAL COACHING",
        "See the problem. See the solution.",
        (
            "The drill is based on the exact tactical situation "
            "identified in your match."
        )
    )


    # ========================================================
    # COURT DIAGRAMS
    # ========================================================

    st.write("")
    st.write("")

    st.caption(
        "TACTICAL TRAINING VISUAL"
    )

    st.header(
        "From attackable → neutral"
    )

    st.write(
        "Follow the situation across three courts: "
        "what happened, what should happen, and how to train it."
    )

    st.write("")

    training_court_diagrams()


    # ========================================================
    # MATCH INTELLIGENCE
    # ========================================================

    st.write("")
    st.write("")

    green_banner(
        "MATCH INTELLIGENCE",
        "This wasn't the only one.",
        (
            "6 other comparable situations were identified "
            "in this match. As more matches are analysed, "
            "Tennis AI can track whether this pattern improves."
        )
    )