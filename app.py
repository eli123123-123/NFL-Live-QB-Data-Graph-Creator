import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st
import nflreadpy as nfl
from matplotlib.ticker import MaxNLocator, MultipleLocator
import os
STRIPE_SECRET = os.getenv("STRIPE_SECRET", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")


# ---------------- constants and helpers ----------------

TEAM_META = {
    "ARI": ("NFC","West"), "ATL": ("NFC","South"), "BAL": ("AFC","North"), "BUF": ("AFC","East"),
    "CAR": ("NFC","South"), "CHI": ("NFC","North"), "CIN": ("AFC","North"), "CLE": ("AFC","North"),
    "DAL": ("NFC","East"), "DEN": ("AFC","West"), "DET": ("NFC","North"), "GB": ("NFC","North"),
    "HOU": ("AFC","South"), "IND": ("AFC","South"), "JAX": ("AFC","South"), "KC": ("AFC","West"),
    "LV": ("AFC","West"), "LAC": ("AFC","West"), "LAR": ("NFC","West"), "MIA": ("AFC","East"),
    "MIN": ("NFC","North"), "NE": ("AFC","East"), "NO": ("NFC","South"), "NYG": ("NFC","East"),
    "NYJ": ("AFC","East"), "PHI": ("NFC","East"), "PIT": ("AFC","North"), "SF": ("NFC","West"),
    "SEA": ("NFC","West"), "TB": ("NFC","South"), "TEN": ("AFC","South"), "WAS": ("NFC","East"),
}

st.cache_data.clear()
st.set_page_config(page_title="NFL Graph Creator", layout="wide")

def normalize_abbr(abbr: str) -> str:
    if not isinstance(abbr, str):
        return ""
    a = abbr.upper()
    aliases = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "JAC": "JAX"}
    return aliases.get(a, a)

@st.cache_data
def load_pbp(seasons):
    pbp_pl = nfl.load_pbp(seasons=seasons)   # pulls latest published nflverse data
    return pbp_pl.to_pandas()

st.session_state["logo_cache"] = {}

def standardize_logo(img: Image.Image, base: int = 100) -> Image.Image:
    img = img.convert("RGBA")
    longest = max(img.size)
    scale = base / longest if longest else 1.0
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGBA", (base, base), (0, 0, 0, 0))
    x = (base - img.width) // 2
    y = (base - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas

@st.cache_data
def fetch_logo_cached(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.content

def fetch_logo(abbr: str):
    ab = normalize_abbr(str(abbr)).strip()
    cache = st.session_state["logo_cache"]
    if ab in cache:
        return cache[ab]
    a = ab.lower()
    urls = [
        f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/logos/teams/{a}.png",
        f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/logos/{a}.png",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/{a}.png",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/transparent/{a}.png",
    ]
    for url in urls:
        try:
            raw = fetch_logo_cached(url)
            img = standardize_logo(Image.open(io.BytesIO(raw)), base=100)
            cache[ab] = img
            return img
        except Exception:
            continue
    cache[ab] = None
    return None

def offset_image(x, y, abbr, ax, zoom=0.12):
    img = fetch_logo(abbr)
    if img is None:
        return
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False, box_alignment=(0.5,0.5))
    ax.add_artist(ab)

def add_trendline(ax, x, y):
    if len(x) < 3:
        return
    xx = np.array(x); yy = np.array(y)
    m, b = np.polyfit(xx, yy, 1)
    xs = np.linspace(xx.min(), xx.max(), 100)
    ax.plot(xs, m*xs + b, linestyle="--", linewidth=1)
    return m, b

# ---------------- title and controls ----------------

st.markdown(
    """
    # NFL Graph Creator  
    <span style="font-size:0.9rem;color:#bbb;">
    Build scatter charts with up to date nflverse play by play data. Pick a position, choose any X and Y metrics, filter weeks and volume, and export a PNG.
    </span>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1,3])
with left:
    seasons = st.multiselect("Seasons", list(range(2009, 2031)), default=[2024])
    week_min, week_max = st.slider("Week range", 1, 22, (1, 18))
    position = st.selectbox("Position", ["QB", "WR", "RB", "TE", "K", "Team Offense", "Team Defense", "Team Overall"])
    show_trend = st.checkbox("Show trend line", value=True)
    show_avg = st.checkbox("Show league average lines", value=True)
    logo_size = st.slider("Logo size", 0.06, 0.40, 0.12)
    show_labels = st.checkbox("Show labels", value=True)
    label_size = st.number_input("Label font size", min_value=6, max_value=16, value=8, step=1)
    confs = st.multiselect("Conference", ["AFC","NFC"])
    divs = st.multiselect("Division", ["East","North","South","West"])

if not seasons:
    st.stop()

# ---------------- data load and features ----------------

pbp = load_pbp(seasons)
pbp = pbp[(pbp["week"] >= week_min) & (pbp["week"] <= week_max)].copy()

# normalize flags
pbp["pass_attempt"]   = (pbp.get("pass_attempt", 0) == 1).astype(int)
pbp["complete_pass"]  = (pbp.get("complete_pass", 0) == 1).astype(int)
pbp["rush_attempt"]   = (pbp.get("rush_attempt", 0) == 1).astype(int)
pbp["dropback"] = (
    (pbp.get("pass", 0) == 1) |
    (pbp.get("sack", 0) == 1) |
    (pbp.get("qb_scramble", 0) == 1) |
    (pbp.get("scramble", 0) == 1)
).astype(int)
pbp["success"]        = (pbp.get("epa", 0.0) > 0).astype(int)
pbp["explosive_pass"] = ((pbp.get("yards_gained", 0) >= 20) & (pbp["pass_attempt"] == 1)).astype(int)
pbp["explosive_rush"] = ((pbp.get("yards_gained", 0) >= 10) & (pbp["rush_attempt"] == 1)).astype(int)
pbp["rz"]             = (pbp.get("yardline_100", 100) <= 20).astype(int)
pbp["third"]          = (pbp.get("down", 0) == 3).astype(int)
pbp["fourth"]         = (pbp.get("down", 0) == 4).astype(int)
pbp["pressure"]       = ((pbp.get("qb_hit", 0) == 1) | (pbp.get("sack", 0) == 1) | (pbp.get("pressure", 0) == 1)).astype(int)

def pick(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return None

cols = set(pbp.columns)
pid_qb   = pick(cols, ["passer_player_id", "passer_id"])
pname_qb = pick(cols, ["passer_player_name", "passer"])
pid_wr   = pick(cols, ["receiver_player_id", "receiver_id"])
pname_wr = pick(cols, ["receiver_player_name", "receiver"])
pid_rb   = pick(cols, ["rusher_player_id", "rusher_id"])
pname_rb = pick(cols, ["rusher_player_name", "rusher"])
pid_k    = pick(cols, ["kicker_player_id", "kicker_id"])
pname_k  = pick(cols, ["kicker_player_name", "kicker"])
team_off = pick(cols, ["posteam", "pos_team"])
team_def = pick(cols, ["defteam", "def_team"])

# ---------------- metrics ----------------

def metric_catalog(pos: str):
    ef = [
        ("Efficiency • EPA per play",                lambda g: g["epa"].mean()),
        ("Efficiency • Success rate",                lambda g: g["success"].mean()),
    ]
    if "cpoe" in cols:
        ef.append(("Efficiency • CPOE",              lambda g: g["cpoe"].mean()))

    vol = [
        ("Volume • Plays",                           lambda g: g["play_id"].count()),
        ("Volume • Attempts",                        lambda g: g["pass_attempt"].sum()),
        ("Volume • Completions",                     lambda g: g["complete_pass"].sum()),
        ("Volume • Rush attempts",                   lambda g: g["rush_attempt"].sum()),
    ]
    yds = [
        ("Yardage • Yards per play",                 lambda g: g["yards_gained"].mean()),
        ("Yardage • Air yards per attempt",          lambda g: g["air_yards"].mean()),
    ]
    exp = [
        ("Explosiveness • Explosive pass rate",      lambda g: g.loc[g["pass_attempt"] == 1, "explosive_pass"].mean()),
        ("Explosiveness • Explosive rush rate",      lambda g: g.loc[g["rush_attempt"] == 1, "explosive_rush"].mean()),
    ]
    sit = [
        ("Situational • Third down pass SR",         lambda g: ((g["third"] == 1) & (g["pass_attempt"] == 1) & (g["success"] == 1)).mean()),
        ("Situational • Red zone SR",                lambda g: g.loc[g["rz"] == 1, "success"].mean()),
        ("Situational • Pressure rate",              lambda g: g["pressure"].mean()),
    ]
    ball = [
        ("Ball security • Interception rate",        lambda g: g.get("interception", pd.Series(index=g.index, dtype=int)).mean()),
        ("Ball security • Fumble rate",              lambda g: g.get("fumble_lost", pd.Series(index=g.index, dtype=int)).mean()),
    ]
    kick = []

    if pos == "QB":
        ef.insert(1, ("Efficiency • EPA per dropback", lambda g: g.loc[g["dropback"] == 1, "epa"].mean()))
        yds.insert(0, ("Yardage • Yards per attempt",  lambda g: g.loc[g["pass_attempt"] == 1, "yards_gained"].mean()))
        vol.insert(1, ("Volume • Dropbacks",           lambda g: g["dropback"].sum()))
    elif pos in ["WR", "TE"]:
        vol.append(("Volume • Targets",                lambda g: ((g["pass_attempt"] == 1) & (g["complete_pass"].isin([0,1]))).sum()))
        yds.append(("Yardage • YAC per reception",     lambda g: g.loc[g["complete_pass"] == 1, "yards_after_catch"].mean()))
    elif pos == "RB":
        yds.insert(0, ("Yardage • Yards per rush",     lambda g: g.loc[g["rush_attempt"] == 1, "yards_gained"].mean()))
    elif pos == "K":
        kick = [
            ("Kicking • FG attempts",                  lambda g: (g.get("field_goal_attempt", 0) == 1).sum()),
            ("Kicking • FG made",                      lambda g: (g.get("field_goal_result", "") == "made").sum()),
            ("Kicking • FG percent",                   lambda g: (g.get("field_goal_result", "") == "made").mean()),
            ("Kicking • XP percent",                   lambda g: (g.get("extra_point_result","") == "good").mean()),
            ("Kicking • Avg FG distance",              lambda g: g.loc[g.get("field_goal_attempt", 0) == 1, "kick_distance"].mean()),
        ]
    elif pos == "Team Defense":
        ef = [
            ("Efficiency • EPA allowed per play",      lambda g: g["epa"].mean()),
            ("Efficiency • Success allowed",           lambda g: g["success"].mean()),
            ("Efficiency • Dropback EPA allowed",      lambda g: g.loc[g["dropback"] == 1, "epa"].mean()),
            ("Efficiency • Rush EPA allowed",          lambda g: g.loc[g["rush_attempt"] == 1, "epa"].mean()),
        ]
        vol = [("Volume • Plays faced",                lambda g: g["play_id"].count())]
        yds = [("Yardage • Yards allowed per play",    lambda g: g["yards_gained"].mean())]
        sit = [("Situational • Sack rate",             lambda g: g.get("sack", pd.Series(index=g.index, dtype=int)).mean())]
        ball = [("Ball security • Takeaway rate",      lambda g: ((g.get("interception", 0) == 1) | (g.get("fumble_lost", 0) == 1)).mean())]
    elif pos == "Team Overall":
        # combine all plays without offense or defense focus
        pass

    # Build organized list for dropdowns
    cats = [ef, vol, yds, exp, sit, ball, kick]
    metrics = []
    for group in cats:
        for label, fn in group:
            metrics.append((label, fn))
    return metrics

def entity_keys(pos: str):
    if pos == "QB":           return (pid_qb, pname_qb, team_off)
    if pos in ["WR","TE"]:    return (pid_wr, pname_wr, team_off)
    if pos == "RB":           return (pid_rb, pname_rb, team_off)
    if pos == "K":            return (pid_k, pname_k, team_off)
    if pos == "Team Offense": return (None, None, team_off)
    if pos == "Team Defense": return (None, None, team_def)
    if pos == "Team Overall": return (None, None, team_off)  # team field reused
    return (None, None, team_off)

def threshold_label_default(pos: str):
    if pos == "QB": return ("Min dropbacks", 100)
    if pos in ["WR","TE"]: return ("Min targets", 30)
    if pos == "RB": return ("Min rush attempts", 50)
    if pos == "K": return ("Min FG attempts", 5)
    return ("Min plays", 200)

def aggregate_for_position(df: pd.DataFrame, pos: str):
    id_col, name_col, t_col = entity_keys(pos)
    if t_col: df = df[~df[t_col].isna()].copy()
    if name_col and name_col in df.columns:
        df = df[~df[name_col].isna()].copy()

    if pos == "Team Defense":
        keys = [t_col]
    elif name_col and id_col:
        keys = [id_col, name_col, t_col]
    else:
        keys = [t_col]

    g = df.groupby(keys, dropna=False)

    base = g.agg(
        plays=("play_id","count"),
        dropbacks=("dropback","sum"),
        attempts=("pass_attempt","sum"),
        completions=("complete_pass","sum"),
        rush_att=("rush_attempt","sum"),
        yards=("yards_gained","sum"),
        epa_sum=("epa","sum"),
    )

    metrics = metric_catalog(pos)
    for label, fn in metrics:
        try:
            base[label] = g.apply(lambda s: fn(s))
        except Exception:
            base[label] = np.nan

    out = base.reset_index()
    if name_col and name_col in out.columns:
        out.rename(columns={name_col: "name", t_col: "team"}, inplace=True)
    else:
        out["name"] = out[t_col]
        out.rename(columns={t_col: "team"}, inplace=True)

    out["conference"] = out["team"].map(lambda t: TEAM_META.get(normalize_abbr(t), (None, None))[0])
    out["division"]   = out["team"].map(lambda t: TEAM_META.get(normalize_abbr(t), (None, None))[1])
    return out, [label for label, _ in metrics]

# ---------------- volume filter and metric dropdowns ----------------

min_label, min_default = threshold_label_default(position)
with left:
    volume_threshold = st.number_input(min_label, min_value=0, value=min_default, step=5)

agg, metric_names = aggregate_for_position(pbp, position)

# apply volume threshold
if position == "QB":
    coln = "Volume • Dropbacks" if "Volume • Dropbacks" in agg.columns else "dropbacks"
    agg = agg[agg[coln] >= volume_threshold]
elif position in ["WR","TE"]:
    coln = "Volume • Targets"
    if coln in agg.columns:
        agg = agg[agg[coln] >= volume_threshold]
elif position == "RB":
    coln = "Volume • Rush attempts" if "Volume • Rush attempts" in agg.columns else "rush_att"
    agg = agg[agg[coln] >= volume_threshold]
elif position == "K":
    coln = "Kicking • FG attempts"
    if coln in agg.columns:
        agg = agg[agg[coln] >= volume_threshold]
else:
    coln = "Volume • Plays" if "Volume • Plays" in agg.columns else "plays"
    agg = agg[agg[coln] >= volume_threshold]

if confs:
    agg = agg[agg["conference"].isin(confs)]
if divs:
    agg = agg[agg["division"].isin(divs)]

# order metric list so most useful appear first per position
priority = {
    "QB": ["Efficiency • EPA per dropback", "Efficiency • Success rate", "Yardage • Yards per attempt"],
    "WR": ["Volume • Targets", "Yardage • YAC per reception", "Explosiveness • Explosive pass rate"],
    "RB": ["Yardage • Yards per rush", "Efficiency • Success rate"],
    "TE": ["Volume • Targets", "Yardage • YAC per reception"],
    "K":  ["Kicking • FG percent", "Kicking • Avg FG distance"],
    "Team Offense": ["Efficiency • EPA per play", "Efficiency • Success rate", "Explosiveness • Explosive pass rate"],
    "Team Defense": ["Efficiency • EPA allowed per play", "Efficiency • Success allowed"],
    "Team Overall": ["Efficiency • EPA per play", "Efficiency • Success rate"],
}
ordered = [m for m in priority.get(position, []) if m in metric_names] + [m for m in metric_names if m not in priority.get(position, [])]

with right:
    x_label = st.selectbox("X metric", ordered)
    y_label = st.selectbox("Y metric", [m for m in ordered if m != x_label] or [x_label])

x_col = x_label
y_col = y_label

# ---------------- data table and plot ----------------

agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=["team", x_col, y_col])
st.caption(f"rows to plot: {len(agg)}")
st.dataframe(agg[["name","team", x_col, y_col]].sort_values(y_col, ascending=False), use_container_width=True)

fig, ax = plt.subplots(figsize=(9,6))
ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.grid(True, alpha=0.3)

count_cols = {"Touchdowns","Interceptions","Volume • Attempts","Volume • Completions",
              "Volume • Plays","Volume • Dropbacks","Volume • Rush attempts",
              "Kicking • FG attempts","Kicking • FG made"}
if x_col in count_cols:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.xaxis.set_minor_locator(MultipleLocator(1))
if y_col in count_cols:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.yaxis.set_minor_locator(MultipleLocator(1))

x = agg[x_col].astype(float); y = agg[y_col].astype(float)
if len(agg):
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmin == xmax: xmin, xmax = xmin - 1.0, xmax + 1.0
    if ymin == ymax: ymin, ymax = ymin - 1.0, ymax + 1.0
    xpad = (xmax - xmin) * 0.12; ypad = (ymax - ymin) * 0.15
    ax.set_xlim(xmin - xpad, xmax + xpad); ax.set_ylim(ymin - ypad, ymax + ypad); ax.margins(0)

if show_avg and len(agg):
    ax.axvline(x.mean(), color="gray", linewidth=1)
    ax.axhline(y.mean(), color="gray", linewidth=1)

miss = set()
for _, r in agg.iterrows():
    abbr = normalize_abbr(str(r["team"]))
    img = fetch_logo(abbr)
    xv = float(r[x_col]); yv = float(r[y_col])
    if img is None:
        miss.add(abbr); ax.plot(xv, yv, marker="o", markersize=6)
    else:
        im = OffsetImage(img, zoom=logo_size)
        ab = AnnotationBbox(im, (xv, yv), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    if show_labels:
        ax.text(xv, yv, str(r["name"]), fontsize=label_size, ha="center", va="bottom",
                color="black", weight="bold", alpha=0.9)

if miss:
    st.caption(f"Missing logos for: {sorted(miss)}")

if show_trend and len(agg) >= 3:
    add_trendline(ax, x, y)

ax.set_title(f"{position} scatter  {', '.join(map(str, seasons))}  weeks {week_min}-{week_max}  {min_label.lower()} {int(volume_threshold)}")

# Save BEFORE rendering (or keep clear_figure=False)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)
png_bytes = buf.getvalue()

st.pyplot(fig, clear_figure=False)

# Download button
st.download_button("Download chart PNG", data=png_bytes, file_name="nfl_graph.png", mime="image/png")


