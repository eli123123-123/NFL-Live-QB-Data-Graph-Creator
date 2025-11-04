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

# ---------- constants and helpers ----------

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
st.set_page_config(page_title="NFL Scatter Studio", layout="wide")

def normalize_abbr(abbr: str) -> str:
    if not isinstance(abbr, str):
        return ""
    a = abbr.upper()
    aliases = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "JAC": "JAX"}
    return aliases.get(a, a)

@st.cache_data
def load_pbp(seasons):
    # nflreadpy returns Polars frames
    pbp_pl = nfl.load_pbp(seasons=seasons)
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
    xx = np.array(x)
    yy = np.array(y)
    m, b = np.polyfit(xx, yy, 1)
    xs = np.linspace(xx.min(), xx.max(), 100)
    ax.plot(xs, m*xs + b, linestyle="--", linewidth=1)
    return m, b

# ---------- UI ----------

st.title("NFL scatter with team logo markers")

left, right = st.columns([1,3])
with left:
    seasons = st.multiselect("Seasons", list(range(2009, 2031)), default=[2024])
    week_min, week_max = st.slider("Week range", 1, 22, (1, 18))
    entity = st.selectbox("Entity", ["QB", "WR", "RB", "TE", "K", "Team Offense", "Team Defense"])
    show_trend = st.checkbox("Show trend line", value=True)
    show_avg = st.checkbox("Show league average lines", value=True)
    logo_size = st.slider("Logo size", 0.06, 0.40, 0.12)
    show_labels = st.checkbox("Show labels", value=True)
    label_size = st.number_input("Label font size", min_value=6, max_value=16, value=8, step=1)
    confs = st.multiselect("Conference", ["AFC","NFC"])
    divs = st.multiselect("Division", ["East","North","South","West"])

if not seasons:
    st.stop()

# ---------- data load and feature flags ----------

pbp = load_pbp(seasons)
pbp = pbp[(pbp["week"] >= week_min) & (pbp["week"] <= week_max)].copy()

# normalize common flags, always present or default zero
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

# helper to pick schema columns
def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
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

# ---------- metric library ----------

def metric_defs(entity_type: str):
    ef = []   # efficiency
    vol = []  # volume
    yds = []  # yardage
    sit = []  # situational
    exp = []  # explosiveness
    ball = [] # ball security
    kick = [] # kicking

    # shared helpers as lambdas on grouped df g
    ef += [
        ("EPA per play",         lambda g: g["epa"].mean()),
        ("Success rate",         lambda g: g["success"].mean()),
    ]
    yds += [
        ("Air yards per attempt",lambda g: g["air_yards"].mean()),
        ("Yards per play",       lambda g: g["yards_gained"].mean()),
    ]
    exp += [
        ("Explosive pass rate",  lambda g: g.loc[g["pass_attempt"] == 1, "explosive_pass"].mean()),
        ("Explosive rush rate",  lambda g: g.loc[g["rush_attempt"] == 1, "explosive_rush"].mean()),
    ]
    sit += [
        ("Third down pass SR",   lambda g: ((g["third"] == 1) & (g["pass_attempt"] == 1) & (g["success"] == 1)).mean()),
        ("Red zone SR",          lambda g: g.loc[g["rz"] == 1, "success"].mean()),
        ("Pressure rate",        lambda g: g["pressure"].mean()),
    ]
    vol += [
        ("Plays",                lambda g: g["play_id"].count()),
        ("Attempts",             lambda g: g["pass_attempt"].sum()),
        ("Completions",          lambda g: g["complete_pass"].sum()),
        ("Rush attempts",        lambda g: g["rush_attempt"].sum()),
    ]
    ball += [
        ("Interception rate",    lambda g: g.get("interception", pd.Series(index=g.index, dtype=int)).mean()),
        ("Fumble rate",          lambda g: g.get("fumble_lost", pd.Series(index=g.index, dtype=int)).mean()),
    ]

    if "cpoe" in cols:
        ef.append(("CPOE", lambda g: g["cpoe"].mean()))

    # entity specific
    if entity_type == "QB":
        ef.append(("EPA per dropback", lambda g: g.loc[g["dropback"] == 1, "epa"].mean()))
        yds.append(("Yards per attempt", lambda g: g.loc[g["pass_attempt"] == 1, "yards_gained"].mean()))
        vol.append(("Dropbacks", lambda g: g["dropback"].sum()))
    elif entity_type in ["WR", "TE"]:
        vol.append(("Targets", lambda g: ((g["pass_attempt"] == 1) & (g["complete_pass"].isin([0,1]))).sum()))
        yds.append(("YAC per reception", lambda g: g.loc[g["complete_pass"] == 1, "yards_after_catch"].mean()))
    elif entity_type == "RB":
        yds.append(("Yards per rush", lambda g: g.loc[g["rush_attempt"] == 1, "yards_gained"].mean()))
    elif entity_type == "K":
        kick += [
            ("FG attempts",  lambda g: (g.get("field_goal_attempt", 0) == 1).sum()),
            ("FG made",      lambda g: (g.get("field_goal_result", "") == "made").sum()),
            ("FG percent",   lambda g: (g.get("field_goal_result", "") == "made").mean()),
            ("XP percent",   lambda g: (g.get("extra_point_result","") == "good").mean()),
            ("Avg FG distance", lambda g: g.loc[g.get("field_goal_attempt", 0) == 1, "kick_distance"].mean()),
        ]
    elif entity_type == "Team Defense":
        # compute allowed metrics using defense team key
        ef = [
            ("EPA allowed per play", lambda g: g["epa"].mean()),
            ("Success allowed",      lambda g: g["success"].mean()),
            ("Dropback EPA allowed", lambda g: g.loc[g["dropback"] == 1, "epa"].mean()),
            ("Rush EPA allowed",     lambda g: g.loc[g["rush_attempt"] == 1, "epa"].mean()),
        ]
        vol = [("Plays faced", lambda g: g["play_id"].count())]
        yds = [("Yards allowed per play", lambda g: g["yards_gained"].mean())]
        sit = [("Sack rate", lambda g: g.get("sack", pd.Series(index=g.index, dtype=int)).mean())]
        ball = [("Takeaway rate", lambda g: ((g.get("interception", 0) == 1) | (g.get("fumble_lost", 0) == 1)).mean())]

    categories = {}
    if ef:   categories["Efficiency"]  = ef
    if vol:  categories["Volume"]      = vol
    if yds:  categories["Yardage"]     = yds
    if exp:  categories["Explosiveness"]= exp
    if sit:  categories["Situational"] = sit
    if ball: categories["Ball security"]= ball
    if kick: categories["Kicking"]     = kick
    return categories

# ---------- grouping by entity ----------

def entity_keys(entity_type: str):
    if entity_type == "QB":
        return (pid_qb, pname_qb, team_off)
    if entity_type in ["WR", "TE"]:
        return (pid_wr, pname_wr, team_off)
    if entity_type == "RB":
        return (pid_rb, pname_rb, team_off)
    if entity_type == "K":
        return (pid_k, pname_k, team_off)
    if entity_type == "Team Offense":
        return (None, None, team_off)
    if entity_type == "Team Defense":
        return (None, None, team_def)
    return (None, None, team_off)

def min_label_and_default(entity_type: str):
    if entity_type == "QB":
        return ("Min dropbacks", 100)
    if entity_type in ["WR", "TE"]:
        return ("Min targets", 30)
    if entity_type == "RB":
        return ("Min rush attempts", 50)
    if entity_type == "K":
        return ("Min FG attempts", 5)
    return ("Min plays", 200)

# ---------- aggregate by entity and compute metrics ----------

def agg_by_entity(df: pd.DataFrame, entity_type: str):
    id_col, name_col, team_col = entity_keys(entity_type)

    # drop null teams
    if team_col:
        df = df[~df[team_col].isna()].copy()

    # drop rows without name when applicable
    if name_col and name_col in df.columns:
        df = df[~df[name_col].isna()].copy()

    if entity_type == "Team Defense":
        key_cols = [team_col]
    elif name_col and id_col:
        key_cols = [id_col, name_col, team_col]
    else:
        key_cols = [team_col]

    g = df.groupby(key_cols, dropna=False)

    # start with base frame with a few simple aggregations for speed
    base = g.agg(
        plays=("play_id", "count"),
        dropbacks=("dropback", "sum"),
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        rush_att=("rush_attempt", "sum"),
        yards=("yards_gained", "sum"),
        epa_sum=("epa", "sum"),
    )

    # compute all metric functions
    cats = metric_defs(entity_type)
    for cat, items in cats.items():
        for label, func in items:
            try:
                base[label] = g.apply(lambda s: func(s))
            except Exception:
                base[label] = np.nan

    out = base.reset_index()

    # unify names for plotting table
    if name_col and name_col in out.columns:
        out.rename(columns={name_col: "name", team_col: "team"}, inplace=True)
    else:
        out["name"] = out[team_col]
        out.rename(columns={team_col: "team"}, inplace=True)

    # attach conference and division
    out["conference"] = out["team"].map(lambda t: TEAM_META.get(normalize_abbr(t), (None, None))[0])
    out["division"]   = out["team"].map(lambda t: TEAM_META.get(normalize_abbr(t), (None, None))[1])

    return out, cats

# ---------- build metric choices and filters ----------

min_label, min_default = min_label_and_default(entity)
with left:
    volume_threshold = st.number_input(min_label, min_value=0, value=min_default, step=5)

# prepare aggregated table
agg, categories = agg_by_entity(pbp, entity)

# apply entity specific volume filter
if entity == "QB":
    agg = agg[agg["Dropbacks"] if "Dropbacks" in agg.columns else agg["dropbacks"] >= volume_threshold]
    if "Dropbacks" in agg.columns:
        agg = agg[agg["Dropbacks"] >= volume_threshold]
    else:
        agg = agg[agg["dropbacks"] >= volume_threshold]
elif entity in ["WR", "TE"]:
    targets_col = "Targets" if "Targets" in agg.columns else None
    if targets_col:
        agg = agg[agg[targets_col] >= volume_threshold]
    else:
        # rough proxy if Targets not present
        agg = agg[agg["Completions"] + agg["Attempts"] >= volume_threshold]
elif entity == "RB":
    ra = "Rush attempts" if "Rush attempts" in agg.columns else "rush_att"
    agg = agg[agg[ra] >= volume_threshold]
elif entity == "K":
    fga = "FG attempts" if "FG attempts" in agg.columns else None
    if fga:
        agg = agg[agg[fga] >= volume_threshold]
else:
    plays_col = "Plays" if "Plays" in agg.columns else "plays"
    agg = agg[agg[plays_col] >= volume_threshold]

# conference and division filters
if confs:
    agg = agg[agg["conference"].isin(confs)]
if divs:
    agg = agg[agg["division"].isin(divs)]

# all metric names available for this entity
all_metrics = []
for items in categories.values():
    for label, _ in items:
        all_metrics.append(label)

# if a metric is not numeric everywhere, coerce
for c in all_metrics:
    if c in agg.columns:
        agg[c] = pd.to_numeric(agg[c], errors="coerce")

with right:
    cat_choice = st.selectbox("Metric category", list(categories.keys()))
    x_label = st.selectbox("X metric", [m for m, _ in categories[cat_choice]])
    # pick a different default for Y if possible
    y_options = [m for m, _ in categories[cat_choice] if m != x_label] or [x_label]
    y_label = st.selectbox("Y metric", y_options)

x_col = x_label
y_col = y_label

# clean and confirm rows
agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=["team", x_col, y_col])
st.caption(f"rows to plot: {len(agg)}")
st.dataframe(agg[["name","team", x_col, y_col]].sort_values(y_col, ascending=False), use_container_width=True)

# ---------- plot ----------

fig, ax = plt.subplots(figsize=(9,6))
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.grid(True, alpha=0.3)

count_cols = {"Touchdowns","Interceptions","Attempts","Completions","Plays","Dropbacks","Rush attempts","FG attempts","FG made"}
if x_col in count_cols:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.xaxis.set_minor_locator(MultipleLocator(1))
if y_col in count_cols:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.yaxis.set_minor_locator(MultipleLocator(1))

x = agg[x_col].astype(float)
y = agg[y_col].astype(float)

if len(agg):
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmin == xmax: xmin, xmax = xmin - 1.0, xmax + 1.0
    if ymin == ymax: ymin, ymax = ymin - 1.0, ymax + 1.0
    xpad = (xmax - xmin) * 0.12
    ypad = (ymax - ymin) * 0.15
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.margins(0)

if show_avg and len(agg):
    ax.axvline(x.mean(), color="gray", linewidth=1)
    ax.axhline(y.mean(), color="gray", linewidth=1)

miss = set()
for _, r in agg.iterrows():
    abbr = normalize_abbr(str(r["team"]))
    img = fetch_logo(abbr)
    xval = float(r[x_col]); yval = float(r[y_col])
    if img is None:
        miss.add(abbr)
        ax.plot(xval, yval, marker="o", markersize=6)
    else:
        im = OffsetImage(img, zoom=logo_size)
        ab = AnnotationBbox(im, (xval, yval), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    if show_labels:
        ax.text(xval, yval, str(r["name"]), fontsize=label_size, ha="center", va="bottom", color="black", weight="bold", alpha=0.9)

if miss:
    st.caption(f"Missing logos for: {sorted(miss)}")

if show_trend and len(agg) >= 3:
    add_trendline(ax, x, y)

title_entity = entity
ax.set_title(f"{title_entity} scatter: {', '.join(map(str, seasons))} weeks {week_min}-{week_max}  {min_label.lower()} {int(volume_threshold)}")
st.pyplot(fig, clear_figure=True)

# ---------- download ----------

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
st.download_button("Download chart PNG", data=buf.getvalue(), file_name="nfl_scatter.png", mime="image/png")
