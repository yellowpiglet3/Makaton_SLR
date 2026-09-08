Download mediapipe_dataset.csv from :

https://drive.google.com/file/d/1BrZRiGV2t8swFcHz6UqR0ma8ckrkmLeR/view?usp=sharing

Place it in the project root:
Makaton_SLR/mediapipe_dataset.csv

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run streamlit_app.py
   ```
