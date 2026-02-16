"""
Streamlit Dashboard for Music Listening EDA
Based on dataset_EDA.ipynb from Lifetime_Albums_TF_Reccomendor
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

import networkx as nx
import plotly.graph_objects as go
from matplotlib.colors import to_hex

from data_loader import load_data, DEFAULT_DATA_DIR
from utils import format_label

# Page config
st.set_page_config(page_title="Music Listening EDA", layout="wide", initial_sidebar_state="expanded")

# Load data with caching
@st.cache_data
def get_data():
    data_dir = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
    if not os.path.exists(os.path.join(data_dir, "recovered_users_data_real.csv")):
        st.error(
            f"Data files not found in {data_dir}. "
            "Place users_data_real.csv, releases_data_real.csv, artists_data_real.csv there, "
            "or set DATA_DIR environment variable."
        )
        return None
    return load_data(data_dir)


def init_session():
    if "data_loaded" not in st.session_state:
        result = get_data()
        if result is not None:
            (
                st.session_state.users_frame,
                st.session_state.albums_frame,
                st.session_state.artists_frame,
            ) = result
            st.session_state.data_loaded = True
        else:
            st.session_state.data_loaded = False


# Sidebar navigation
st.sidebar.title("ListenBrainz Dataset EDA")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Genre Analysis Per Year", "Genre Networks"],
    label_visibility="collapsed",
)

init_session()

if not st.session_state.get("data_loaded", False):
    st.warning("Load data first. Ensure CSV files are in the data directory.")
    st.stop()

users_frame = st.session_state.users_frame
albums_frame = st.session_state.albums_frame
artists_frame = st.session_state.artists_frame



# --- Home Page ---
if page == "Home":
    st.title("Home")
    st.markdown("Before training my reccomender models for the final [Album-Per-Year project](https://github.com/RishmitaR/Album-Per-Year) I did exploratory data analysis on the dataset I created from the API calls I made to ListenBrainz. This dashboard summarizes interesting insights from that EDA and these visualizations should be helpful when modelling the reccomender systems. The original dataset had 42,000 user profiles, but I filtered out users with fewer than 1000 listens to focus on more active listeners. The data includes user artist and albums listening histories, as well as genre information for artists and albums pulled from the MusicBrainz database where available. Release Years were also included for albums.")
    st.markdown("### Overview")

    listens = users_frame["listen_count"]
    mean = np.mean(listens)
    q1 = np.percentile(listens, 25)
    median = np.median(listens)
    q3 = np.percentile(listens, 75)
    iqr = q3 - q1
    lower_whisker = listens[listens >= q1 - 1.5 * iqr].min()
    upper_whisker = listens[listens <= q3 + 1.5 * iqr].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Users", f"{len(users_frame):,}")
    col2.metric("Number of Artists in Database", f"{artists_frame["artist_name"].nunique():,}")
    col3.metric("Number of Albums in Database", f"{albums_frame["release_name"].nunique():,}")
    col4.metric("Mean Song Listens", f"{mean:,.0f}")
    col5.metric("Median Song Listens", f"{int(median):,}")

    st.markdown("### Distribution of Song Listens")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(
        x=listens,
        ax=ax,
        showmeans=True,
        meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="black"),
    )
    y = 0
    ax.text(mean, y + 0.12, f"Mean: {mean:.1f}", ha="center", color="red")
    ax.text(q1, y - 0.18, f"Q1: {int(q1)}", ha="center")
    ax.text(median, y - 0.05, f"Median: {int(median)}", ha="center")
    ax.text(q3, y - 0.18, f"Q3: {int(q3)}", ha="center")
    ax.text(lower_whisker, y + 0.12, f"Min*: {int(lower_whisker)}", ha="center")
    ax.text(upper_whisker, y + 0.12, f"Max*: {int(upper_whisker)}", ha="center")
    ax.text(upper_whisker, y - 0.22, f"Absolute Max*: {int(np.max(listens))}", ha="center")
    padding = 0.05 * (upper_whisker - lower_whisker)
    ax.set_xlim(left=lower_whisker - padding, right=upper_whisker + padding)
    ax.set_xlabel("Song Listens")
    ax.set_title("Distribution of Song Listens")
    st.pyplot(fig)
    plt.close()

    # Distribution of album release years (one entry per unique album)
    st.markdown("### Distribution of Album Release Years")
    albums_unique = albums_frame.drop_duplicates(subset=["release_name", "artist_name"]).copy()
    album_years = albums_unique["release_year"].dropna().astype(int)
    if len(album_years) > 0:
        fig_alb, ax_alb = plt.subplots(figsize=(14, 4))
        fig_alb.patch.set_facecolor('white')
        ax_alb.set_facecolor('white')
        sns.boxplot(
            x=album_years,
            ax=ax_alb,
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="black"),
        )
        mean_a = np.mean(album_years)
        q1_a = np.percentile(album_years, 25)
        median_a = np.median(album_years)
        q3_a = np.percentile(album_years, 75)
        iqr_a = q3_a - q1_a
        lower_whisker_a = album_years[album_years >= q1_a - 1.5 * iqr_a].min()
        upper_whisker_a = album_years[album_years <= q3_a + 1.5 * iqr_a].max()
        y_a = 0
        ax_alb.text(mean_a, y_a + 0.12, f"Mean: {mean_a:.0f}", ha="center", color="red")
        ax_alb.text(q1_a, y_a - 0.18, f"Q1: {int(q1_a)}", ha="center")
        ax_alb.text(median_a, y_a - 0.05, f"Median: {int(median_a)}", ha="center")
        ax_alb.text(q3_a, y_a - 0.18, f"Q3: {int(q3_a)}", ha="center")
        ax_alb.text(lower_whisker_a, y_a + 0.12, f"Min*: {int(lower_whisker_a)}", ha="center")
        ax_alb.text(lower_whisker_a, y_a - 0.22, f"Absolute Min*: {int(np.min(album_years))}", ha="center")
        ax_alb.text(upper_whisker_a, y_a + 0.12, f"Max*: {int(upper_whisker_a)}", ha="center")
        ax_alb.text(upper_whisker_a, y_a - 0.22, f"Absolute Max*: {int(np.max(album_years))}", ha="center")
        padding_a = 0.25 * (upper_whisker_a - lower_whisker_a) if upper_whisker_a != lower_whisker_a else 5
        ax_alb.set_xlim(left=lower_whisker_a - padding_a, right=upper_whisker_a + padding_a)
        ax_alb.set_xlabel("Release Year")
        ax_alb.set_title("Distribution of Album Release Years (per unique album)")
        plt.tight_layout()
        st.pyplot(fig_alb)
        plt.close()
    
    # Top artists and albums (moved to Home; global, no year filter)
    n_artists = st.slider("Top Artists to Show", min_value=5, max_value=50, value=15, key="home_artists_n")
    n_albums = st.slider("Top Albums to Show", min_value=5, max_value=50, value=15, key="home_albums_n")

    # prepare palettes
    cmap = plt.cm.get_cmap("tab20c")
    cmap_colors = [to_hex(cmap(i)) for i in range(cmap.N)]

    top_artists = artists_frame["artist_name"].value_counts().head(n_artists)
    top_albums = albums_frame["release_name"].value_counts().head(n_albums)

    artists_df = top_artists.reset_index()
    artists_df.columns = ["Artist", "Count"]
    albums_df = top_albums.reset_index()
    albums_df.columns = ["Album", "Count"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top Artists")
        fig_a, ax_a = plt.subplots(figsize=(8, max(5, n_artists * 0.4)))
        fig_a.patch.set_facecolor('white')
        ax_a.set_facecolor('white')
        sns.barplot(data=artists_df, y="Artist", x="Count", ax=ax_a, palette=cmap_colors[: len(artists_df)])
        ax_a.set_ylabel("Artist")
        ax_a.set_xlabel("Artist Occurrences in User Histories")
        ax_a.set_title(f"Top {n_artists} Artists")
        plt.tight_layout()
        st.pyplot(fig_a)
        plt.close()

    with col2:
        st.markdown("### Top Albums")
        fig_b, ax_b = plt.subplots(figsize=(8, max(5, n_albums * 0.4)))
        fig_b.patch.set_facecolor('white')
        ax_b.set_facecolor('white')
        sns.barplot(data=albums_df, y="Album", x="Count", ax=ax_b, palette=cmap_colors[: len(albums_df)])
        ax_b.set_ylabel("Album")
        ax_b.set_xlabel("Album Occurrences in User Histories")
        ax_b.set_title(f"Top {n_albums} Albums")
        plt.tight_layout()
        st.pyplot(fig_b)
        plt.close()

    st.markdown(" ### Note on Top Artists and Top Albums:")
    st.markdown("These are the top albums and artists in the dataset based on how many times they appear in the listening histories of all users. This is not a total count of all listens of that album or artist, but rather a count of how many users have that album or artist in their listening history. Rather than keeping track of the total listens for each album or artist per user I assigned a weight to each album and artist for each user based on how much of their total listening history that album or artist made up, so this is a count of how many users had that album or artist in their history regardless of how much they listened to it. The genre analysis page also uses this same method of counting genres based on how many users had albums with that genre in their listening history. If I were to redo the ETL of this dataset I would have also saved the total listens for each album and artist per user, which would allow for more detailed analysis of the most listened to albums and artists in the dataset, but I think this method of counting how many users had an album or artist in their history is still a useful way to get a sense of the most popular albums and artists in the dataset without being skewed by a few users with very high listen counts for certain albums or artists. Whaty I've done should also be good enough for recommender system modelling and evaluation, since the proportional presence of an album or artist in a user's history is more important for training and evaluating the models than the total listens.")

# --- Genre Analysis Page ---
elif page == "Genre Analysis Per Year":
    st.title("Genre Analysis Per Year")
    st.markdown("Explore genre popularity by year range.")

    year_min = int(albums_frame["release_year"].min())
    year_max = int(albums_frame["release_year"].max())
    # default starting year should be 1960 if data allows
    start_default = max(year_min, 1975)
    start_year = st.number_input(
        "Starting Year", min_value=year_min, max_value=year_max, value=start_default
    )
    end_year = st.number_input("Ending Year", min_value=year_min, max_value=year_max, value=year_max)
    n_genres = st.slider("Number of Genres to Show", min_value=5, max_value=50, value=20)

    if start_year > end_year:
        st.error("Starting year must be less than or equal to ending year.")
        st.stop()

    filtered = albums_frame.query("release_year >= @start_year and release_year <= @end_year")

    # Aggregate genre counts for the year range
    genre_dict = {}
    genres = albums_frame["genres"]
    for row in genres:
        for genre in row:
            genre_dict[genre] = genre_dict.get(genre, 0) + 1

    genre_count_df = (
        pd.DataFrame(list(genre_dict.items()), columns=["Genre", "Count"])
        .sort_values(by="Count", ascending=False)
        .head(n_genres)
    )
    genre_count_df["Genre Display"] = genre_count_df["Genre"].apply(format_label)

    # Genre counts per year for heatmap
    years = list(range(start_year, end_year + 1))
    genre_year_data = {}

    for g in genre_count_df["Genre"].tolist():
        genre_year_data[g] = {}
        for y in years:
            year_df = filtered[filtered["release_year"] == y]
            count = year_df["genres"].apply(lambda genres: g in genres).sum()
            genre_year_data[g][y] = count

    heatmap_df = pd.DataFrame(genre_year_data).T
    heatmap_df.index = [format_label(g) for g in heatmap_df.index]


    st.markdown("### Top Genres by occurrence in Album Histories")
    st.markdown("As with the top artists and albums, this counts how many times each genre appears in the album listening histories of all users in the selected year range. This is not a count of unique albums or artists with that genre, but rather a count of how many times albums with that genre were listened to across all users.")
    fig1, ax1 = plt.subplots(figsize=(10, max(6, n_genres * 0.55)))
    fig1.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    cmap = plt.cm.get_cmap("tab20c")
    cmap_colors = [to_hex(cmap(i)) for i in range(cmap.N)]
    sns.barplot(
        data=genre_count_df,
        y="Genre Display",
        x="Count",
        ax=ax1,
        palette=cmap_colors[: len(genre_count_df)],
    )
    ax1.set_ylabel("Genre")
    ax1.set_xlabel("Genre Occurrences in Album Histories")
    ax1.set_title(f"Top {n_genres} Genres ({start_year}–{end_year})")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown("### Genre Presence by Year (Heatmap)")
    # make heatmap larger for wide year ranges; increase vertical size with number of years
    fig2_width = max(12, len(years) * 0.6)
    fig2_height = max(8, n_genres * 0.6, len(years) * 0.25)
    fig2, ax2 = plt.subplots(figsize=(fig2_width, fig2_height))
    sns.heatmap(heatmap_df, ax=ax2, cmap="YlOrRd", annot=False, fmt="d")
    ax2.set_title(f"Genre Listens per Year ({start_year}–{end_year})")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Genre")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# --- Artists and Albums Page ---
# Note: the previous "Artists and Albums" page was moved to Home and removed.


# --- Genre Networks Page ---
elif page == "Genre Networks":
    st.title("Genre Networks")
    st.markdown("Genre co-occurrence networks by Louvian communities. Genres are connected if they co-occur in the same album, with edge weight based on how many albums had that genre pair. Louvain communities are detected and the strongest genre in each community is highlighted in red. The bar graph shows the most common genre co-occurrences within that community.")

    try:
        import community as community_louvain
    except ImportError:
        st.error(
            "Please install python-louvain: `pip install python-louvain`"
        )
        st.stop()

    @st.cache_data
    def build_genre_network():
        pairs = Counter()
        for _artist_name, group in artists_frame.groupby("artist_name"):
            genres = group["genres"].iloc[0]
            for g1, g2 in combinations(sorted(genres), 2):
                pairs[(g1, g2)] += 1
        edges_df = pd.DataFrame(
            [(g1, g2, w) for (g1, g2), w in pairs.items()],
            columns=["Source", "Target", "Weight"],
        )
        G = nx.from_pandas_edgelist(
            edges_df, source="Source", target="Target", edge_attr="Weight"
        )
        partition = community_louvain.best_partition(
            G, weight="Weight", resolution=1.0, random_state=42
        )
        strongest_nodes = {}
        for c in set(partition.values()):
            nodes_in_comm = [n for n, comm in partition.items() if comm == c]
            G_comm = G.subgraph(nodes_in_comm)
            top_node = max(G_comm.degree(weight="Weight"), key=lambda x: x[1])[0]
            strongest_nodes[c] = top_node
        return G, partition, strongest_nodes, pairs

    G, partition, strongest_nodes, pairs = build_genre_network()
    communities = sorted(set(partition.values()))
    cmap = plt.cm.get_cmap("tab20c")
    community_map = {c: cmap(i % 20) for i, c in enumerate(communities)}

    for comm_id in communities:
        community_nodes = [n for n, c in partition.items() if c == comm_id]
        if len(community_nodes) <= 2:
            continue
        strongest = strongest_nodes[comm_id]
        st.markdown(f"### {format_label(strongest)}")

        G_sub = G.subgraph(community_nodes).copy()
        top_nodes = sorted(
            G_sub.degree(weight="Weight"),
            key=lambda x: x[1],
            reverse=True,
        )[:65]
        G_sub = G_sub.subgraph([n for n, _ in top_nodes]).copy()

        # Network graph (shown first) – Plotly for translucent labels and hover
        pos_sub = nx.spring_layout(G_sub, k=50.0, seed=42, iterations=1000, weight="Weight")
        node_colors = [community_map[partition[n]] for n in G_sub.nodes()]
        node_sizes = [G_sub.degree(n, weight="Weight") * 0.5 for n in G_sub.nodes()]
        edge_weights = [d["Weight"] for _, _, d in G_sub.edges(data=True)]
        max_w = max(edge_weights) if edge_weights else 1
        node_list = list(G_sub.nodes())

        # Edge trace
        edge_x, edge_y = [], []
        for (u, v, d) in G_sub.edges(data=True):
            x0, y0 = pos_sub[u]
            x1, y1 = pos_sub[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=1, color="rgba(150,150,150,0.3)"),
            hoverinfo="none",
        )

        # Node positions and styling
        node_x = [pos_sub[n][0] for n in node_list]
        node_y = [pos_sub[n][1] for n in node_list]
        node_color_hex = [to_hex(c) for c in node_colors]
        # Scale sizes for Plotly (pixel units); keep relative scale
        node_size_vals = [min(40, max(8, s / 2)) for s in node_sizes]
        strongest_idx = node_list.index(strongest)
        node_color_hex[strongest_idx] = "red"
        node_size_vals[strongest_idx] = min(50, max(12, G_sub.degree(strongest, weight="Weight") * 0.3))
        labels = [format_label(n) for n in node_list]

        # Node trace (markers only; hover shows genre name larger)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(size=node_size_vals, color=node_color_hex, line=dict(width=0)),
            text=labels,
            hoverinfo="text",
            hoverlabel=dict(
                font=dict(size=16, family="sans-serif", color="black"),
                bgcolor="white",
                bordercolor="white",
            ),
            name="",
        )

        # Text labels (slightly translucent)
        text_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="text",
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="rgba(0,0,0,0.65)", family="sans-serif"),
            hoverinfo="none",
            name="",
        )

        fig_net = go.Figure(data=[edge_trace, node_trace, text_trace])
        fig_net.update_layout(
            title=dict(text=f"Genre Network – {format_label(strongest)}", x=0.5, xanchor="center"),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=500,
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Bar graph of most common genre co-occurrences in this community
        comm_pairs = Counter()
        for (g1, g2), w in pairs.items():
            if g1 in G_sub.nodes and g2 in G_sub.nodes:
                label = f"{format_label(g1)} – {format_label(g2)}"
                comm_pairs[label] += w
        top_pairs = comm_pairs.most_common(15)
        if top_pairs:
            pair_df = pd.DataFrame(top_pairs, columns=["Genre Pair", "Count"])
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            fig_bar.patch.set_facecolor('white')
            ax_bar.set_facecolor('white')
            cmap = plt.cm.get_cmap("tab20c")
            cmap_colors = [to_hex(cmap(i)) for i in range(cmap.N)]
            sns.barplot(data=pair_df, y="Genre Pair", x="Count", ax=ax_bar, palette=cmap_colors[: len(pair_df)])
            ax_bar.set_title(f"Top Genre Co-occurrences – {format_label(strongest)}")
            ax_bar.set_ylabel("Genre Pair")
            ax_bar.set_xlabel("Co-occurrence Count")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()
    st.markdown("### Note on Genre Networks:")
    st.markdown("I did this part of the EDA purely out of curiosity, and I'm not sure how useful it will be for the reccomender. Maybe this is something that will be done under the hood while training? This network and louvian communities might be helpful to reccomend albums from external databases that match nicher genre occurances that occur in this dataset but dont have listens in the ListenBrainz dataset.")

