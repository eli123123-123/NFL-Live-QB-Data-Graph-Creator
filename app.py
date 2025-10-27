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



TEAM_META = {
 "ARI": ("NFC","West"), "ATL": ("NFC","South"), "BAL": ("AFC","North"), "BUF": ("AFC","East"),
 "CAR": ("NFC","South"), "CHI": ("NFC","North"), "CIN": ("AFC","North"), "CLE": ("AFC","North"),
 "DAL": ("NFC","East"),  "DEN": ("AFC","West"),  "DET": ("NFC","North"), "GB":  ("NFC","North"),
 "HOU": ("AFC","South"), "IND": ("AFC","South"), "JAX": ("AFC","South"), "KC":  ("AFC","West"),
 "LV":  ("AFC","West"),  "LAC": ("AFC","West"),  "LAR": ("NFC","West"),  "MIA": ("AFC","East"),
 "MIN": ("NFC","North"), "NE":  ("AFC","East"),  "NO":  ("NFC","South"), "NYG": ("NFC","East"),
 "NYJ": ("AFC","East"),  "PHI": ("NFC","East"),  "PIT": ("AFC","North"), "SF":  ("NFC","West"),
 "SEA": ("NFC","West"),  "TB":  ("NFC","South"),  "TEN": ("AFC","South"), "WAS": ("NFC","East"),
}

# clear any stale cache from earlier wrong logo paths
st.cache_data.clear()


st.set_page_config(page_title="QB Chart Maker", layout="wide")

def normalize_abbr(abbr: str) -> str:
    if not isinstance(abbr, str):
        return ""
    a = abbr.upper()
    aliases = {
        "LA": "LAR", "STL": "LAR",
        "SD": "LAC",
        "OAK": "LV",
        "WSH": "WAS",
        "JAC": "JAX",
    }
    return aliases.get(a, a)

@st.cache_data
def load_pbp(seasons):
    pbp_pl = nfl.load_pbp(seasons=seasons)   # Polars DataFrame
    return pbp_pl.to_pandas()                # convert to pandas for the rest of your code



# always reset to avoid stale None entries from earlier bad URLs
st.session_state["logo_cache"] = {}
def standardize_logo(img: Image.Image, base: int = 100) -> Image.Image:
    """Make every logo square but keep high resolution so zoom slider works."""
    img = img.convert("RGBA")
    longest = max(img.size)
    # scale so longest side = base px
    scale = base / longest
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, Image.LANCZOS)
    # center on square canvas
    canvas = Image.new("RGBA", (base, base), (0, 0, 0, 0))
    x = (base - img.width) // 2
    y = (base - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


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
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                raw = Image.open(io.BytesIO(r.content))
                img = standardize_logo(raw, base=100)
                cache[ab] = img
                return img
        except Exception:
            pass

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

st.title("QB scatter with team logo markers")

# Controls
left, right = st.columns([1,3])
with left:
    seasons = st.multiselect("Seasons", list(range(2010, 2031)), default=[2024])
    week_min, week_max = st.slider("Week range", 1, 22, (1, 18))
    min_dropbacks = st.number_input("Min dropbacks", min_value=1, value=100, step=10)
    group_by = st.selectbox("Group at QB or Team level", ["QB", "Team"])
    show_trend = st.checkbox("Show trend line", value=True)
    show_avg = st.checkbox("Show league average lines", value=True)
    logo_size = st.slider("Logo size", 0.06, 0.4, 0.12)
    show_labels = st.checkbox("Show QB labels", value=True)
label_size = st.number_input("Label font size", min_value=6, max_value=14, value=8, step=1)
confs = st.multiselect("Conference", ["AFC","NFC"])
divs = st.multiselect("Division", ["East","North","South","West"])


if not seasons:
    st.stop()

pbp = load_pbp(seasons)
pbp = pbp[(pbp["week"] >= week_min) & (pbp["week"] <= week_max)]

# make dropback column robust to missing scramble fields
pbp["dropback"] = (
    (pbp.get("pass", 0) == 1) |
    (pbp.get("sack", 0) == 1) |
    (pbp.get("qb_scramble", 0) == 1) |
    (pbp.get("scramble", 0) == 1)
).astype(int)

# detect actual QB and team columns in current dataset
pid_col = "passer_id" if "passer_id" in pbp.columns else ("passer_player_id" if "passer_player_id" in pbp.columns else None)
pname_col = "passer" if "passer" in pbp.columns else ("passer_player_name" if "passer_player_name" in pbp.columns else None)
team_col = "posteam" if "posteam" in pbp.columns else ("pos_team" if "pos_team" in pbp.columns else None)
# keep only players whose roster position is QB
if pid_col:
  roster_pl = nfl.load_rosters_weekly(seasons=seasons)
  roster_df = roster_pl.to_pandas()

# Try to find sensible id/position columns across seasons/vendors
id_col  = next((c for c in ["player_id", "gsis_id", "nfl_id"] if c in roster_df.columns), None)
pos_col = next((c for c in ["position", "pos"] if c in roster_df.columns), None)

if id_col and pos_col and pid_col:
    roster = roster_df[[id_col, pos_col]].drop_duplicates(id_col).copy()
    roster.columns = ["player_id", "position"]  # normalize names
    pbp = pbp.merge(roster, left_on=pid_col, right_on="player_id", how="left")
    pbp = pbp[pbp["position"] == "QB"].drop(columns=["player_id", "position"])
# else: fall back to no roster-based QB filter if columns aren't present

# drop invalid rows
if team_col:
    pbp = pbp[~pbp[team_col].isna()]
if group_by == "QB" and pid_col and pname_col:
    pbp = pbp[~pbp[pid_col].isna() & ~pbp[pname_col].isna()]

# pick grouping level
if group_by == "QB" and pid_col and pname_col:
    grp_cols = [pid_col, pname_col, team_col]
else:
    grp_cols = [team_col]

agg = pbp.groupby(grp_cols).agg(
    plays=("play_id","count"),
    dropbacks=("dropback","sum"),
    attempts=("pass_attempt","sum"),
    completions=("complete_pass","sum"),
    yards=("yards_gained","sum"),
    tds=("pass_touchdown","sum"),
    ints=("interception","sum"),
    epa_sum=("epa","sum"),
    epa_per_play=("epa","mean"),
    success_rate=("epa", lambda s: np.mean(s > 0)),
    cpoe=("cpoe","mean"),
    air_yards=("air_yards","mean"),
    ypa=("yards_gained","mean"),
).reset_index()
agg = agg[agg["dropbacks"] >= min_dropbacks]
# rename for uniform output
if group_by == "QB" and pname_col:
    agg.rename(columns={pname_col: "name", team_col: "team"}, inplace=True)
else:
    agg["name"] = agg[team_col]
    agg.rename(columns={team_col: "team"}, inplace=True)
# attach conference and division metadata
agg["conference"] = agg["team"].map(lambda t: TEAM_META.get(normalize_abbr(t), (None, None))[0])
agg["division"]   = agg["team"].map(lambda t: TEAM_META.get(normalize_abbr(t), (None, None))[1])

# apply filters if user selected any
if confs:
    agg = agg[agg["conference"].isin(confs)]
if divs:
    agg = agg[agg["division"].isin(divs)]


# Choose metrics
metric_options = {
    "EPA per play": "epa_per_play",
    "Success rate": "success_rate",
    "CPOE": "cpoe",
    "Yards per play": "ypa",
    "Total EPA": "epa_sum",
    "Touchdowns": "tds",
    "Interceptions": "ints",
    "Attempts": "attempts",
    "Completions": "completions",
    "Air yards": "air_yards",
}

with right:
    c1, c2 = st.columns(2)
    with c1:
        x_label = st.selectbox("X metric", list(metric_options.keys()), index=0)
    with c2:
        y_label = st.selectbox("Y metric", list(metric_options.keys()), index=1)

x_col = metric_options[x_label]
y_col = metric_options[y_label]

# clean and confirm rows
agg = agg.replace([np.inf, -np.inf], np.nan)
agg = agg.dropna(subset=[x_col, y_col, "team"])
st.write(f"rows to plot: {len(agg)}")
st.dataframe(agg[["name","team", x_col, y_col]].head(15), use_container_width=True)

# Figure
fig, ax = plt.subplots(figsize=(9,6))
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.grid(True, alpha=0.3)

# Force integer ticks for count metrics  <-- paste here
count_cols = {"tds", "ints", "attempts", "completions", "plays", "dropbacks"}
if x_col in count_cols:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
if y_col in count_cols:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

# Axis means
x = agg[x_col].astype(float)
y = agg[y_col].astype(float)

# Expand axes so logos/labels are inside the frame
xmin, xmax = float(x.min()), float(x.max())
ymin, ymax = float(y.min()), float(y.max())

# guard against zero-range axes
if xmin == xmax:
    xmin -= 1.0
    xmax += 1.0
if ymin == ymax:
    ymin -= 1.0
    ymax += 1.0

xpad = (xmax - xmin) * 0.12   # horizontal padding (~12%)
ypad = (ymax - ymin) * 0.15   # vertical padding (~15%)

ax.set_xlim(xmin - xpad, xmax + xpad)
ax.set_ylim(ymin - ypad, ymax + ypad)
ax.margins(0)  # we’re managing padding manually

# League averages
if show_avg:
    ax.axvline(x.mean(), color="gray", linewidth=1)
    ax.axhline(y.mean(), color="gray", linewidth=1)

st.write("logo cache size:", len(st.session_state["logo_cache"]))

# Plot logos
plotted = 0
miss = set()
for _, r in agg.iterrows():
    abbr = normalize_abbr(str(r["team"]))
    img = fetch_logo(abbr)
    xval = float(r[x_col])
    yval = float(r[y_col])

    if img is None:
        miss.add(abbr)
        ax.plot(xval, yval, marker="o", markersize=6)
    else:
        im = OffsetImage(img, zoom=logo_size)
        ab = AnnotationBbox(im, (xval, yval), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

    # QB name label
    if show_labels:
        ax.text(
        xval, yval,
        str(r["name"]),
        fontsize=label_size,
        ha="center", va="bottom",
        color="black",   # changed from white to black
        weight="bold",
        alpha=0.9,
    )



    

if miss:
    st.caption(f"Missing logos for: {sorted(miss)}")


# Trend line
if show_trend:
    add_trendline(ax, x, y)

ax.set_title(f"{group_by} scatter: {', '.join(map(str, seasons))} weeks {week_min}-{week_max}  min dropbacks {int(min_dropbacks)}")
st.pyplot(fig, clear_figure=True)

# Data table and download
st.dataframe(agg.sort_values(y_col, ascending=False), use_container_width=True)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
st.download_button("Download chart PNG", data=buf.getvalue(), file_name="qb_scatter.png", mime="image/png")






