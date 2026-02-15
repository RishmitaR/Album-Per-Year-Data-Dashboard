# Music Listening EDA Dashboard

Streamlit dashboard for exploring music listening data (users, artists, albums, genres).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the following CSV files in `../Lifetime_Albums_TF_Reccomendor/` (relative to this project):
   - `users_data_real.csv`
   - `releases_data_real.csv`
   - `artists_data_real.csv`

   Or set the `DATA_DIR` environment variable to your data folder:
   ```bash
   export DATA_DIR=/path/to/your/data
   streamlit run app.py
   ```

## Run

```bash
streamlit run app.py
```

## Pages

- **Home**: Summary statistics and boxplot of song listens distribution
- **Genre Analysis**: Year range filter, horizontal bar plot of top genres, heatmap of genres by year
- **Artists and Albums**: Top artists and albums with year range and top-n filters
- **Genre Networks**: Louvain community detection on genre co-occurrence graph; each community shows a bar of co-occurrences and a network visualization
