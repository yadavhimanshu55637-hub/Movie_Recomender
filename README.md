# Movie Recommendation System

A simple movie recommendation project using content-based filtering and TF-IDF on movie descriptions.

## What is included

- `app2.py`: runnable Python script with recommendation and search helpers
- `requirements.txt`: required Python packages
- `data/movies.csv`: sample movie dataset

## Setup

1. Create a Python virtual environment if needed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
pip install -r requirements.txt
streamlit run app2.py
```

> Ignore `streamlit_app.py` — the active Streamlit interface is in `app2.py`.

## Extend

- Add more rows to `data/movies.csv`
- Replace the content-based engine with collaborative filtering
- Build a web interface using Flask or Streamlit
