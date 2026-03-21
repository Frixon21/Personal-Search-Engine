# Personal Search Engine

Local file search backed by SQLite. The project can index a folder, search it with lexical, semantic, or hybrid ranking, and expose the same backend through a small FastAPI web UI.

## Install

Use the project virtual environment or create one, then install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run The Web App

Start the FastAPI app with Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m uvicorn pse_web.app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

If the repo contains a `Files` folder, the web UI uses it as the default search root.

## Project Layout

- `pse_web/app.py`
  FastAPI entrypoint, routes, template rendering, and static asset mounting.
- `pse_web/services.py`
  View-model helpers that format backend data for the templates.
- `templates/`
  Jinja2 templates for the page, search results, and indexing status.
- `static/`
  Small custom stylesheet. Tailwind CSS and HTMX are loaded from CDNs so there is no frontend build step.

## Backend

The web UI uses the existing backend code:

- Indexing still runs through `pse_index.index_root(...)`
- Search still runs through `pse_search`
- CLI commands still work as before

## CLI

Index a folder:

```powershell
.\.venv\Scripts\python.exe pse_index.py index .\Files
```

Search from the CLI:

```powershell
.\.venv\Scripts\python.exe pse_search.py .\Files "project sync recap" 10
```
