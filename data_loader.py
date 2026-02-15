"""Load and process music listening data."""

import os
import pandas as pd
from utils import clean_genres

# Default data path - current directory (where data_loader.py lives)
DEFAULT_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(data_dir=None):
    """Load users, albums, and artists data with clean genre lists."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    users_path = os.path.join(data_dir, "recovered_users_data_real.csv")
    releases_path = os.path.join(data_dir, "recovered_releases_data_real.csv")
    artists_path = os.path.join(data_dir, "recovered_artists_data_real.csv")    

    users_frame = pd.read_csv(users_path)
    albums_frame = pd.read_csv(releases_path)
    artists_frame = pd.read_csv(artists_path)

    artists_frame["genres_list"] = artists_frame["genres"].apply(clean_genres)
    albums_frame["genres_list"] = albums_frame["genres"].apply(clean_genres)

    artists_with_genres = artists_frame[artists_frame["genres_list"].apply(lambda x: len(x) > 0)]
    albums_with_genres = albums_frame[albums_frame["genres_list"].apply(lambda x: len(x) > 0)]

    final_artists_df = (
        artists_with_genres.groupby(["user_id", "artist_name"], as_index=False)
        .agg(
            weight=("artist_weight", "max"),
            genres=("genres_list", lambda x: max(x, key=len)),
        )
    )

    final_albums_df = (
        albums_with_genres.groupby(
            ["user_id", "release_name", "artist_name", "release_year"], as_index=False
        )
        .agg(
            weight=("release_weight", "max"),
            genres=("genres_list", lambda x: max(x, key=len)),
        )
    )

    return users_frame, albums_frame, artists_frame, final_artists_df, final_albums_df
