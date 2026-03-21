from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from pse_search import SEARCH_MODES
from pse_web import services


BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Personal Search Engine")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request", "").lower() == "true"


def _top_k_value(raw: int | None) -> int:
    if raw is None:
        return services.DEFAULT_TOP_K
    return raw if raw in services.TOP_K_OPTIONS else services.DEFAULT_TOP_K


def _page_context(request: Request, *, search, summary, index_result=None) -> dict[str, object]:
    return {
        "request": request,
        "search": search,
        "summary": summary,
        "index_result": index_result,
        "mode_options": SEARCH_MODES,
        "top_k_options": services.TOP_K_OPTIONS,
    }


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    root: str | None = None,
    mode: str = "hybrid",
    top_k: int = services.DEFAULT_TOP_K,
) -> HTMLResponse:
    resolved_root = root or str(services.get_default_root())
    top_k_value = _top_k_value(top_k)
    search = services.search_documents(resolved_root, "", mode, top_k_value)
    summary = services.get_index_summary(resolved_root)
    return templates.TemplateResponse(
        request,
        "index.html",
        _page_context(request, search=search, summary=summary),
    )


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    root: str | None = None,
    query: str = "",
    mode: str = "hybrid",
    top_k: int = services.DEFAULT_TOP_K,
) -> HTMLResponse:
    resolved_root = root or str(services.get_default_root())
    top_k_value = _top_k_value(top_k)
    search_view = await run_in_threadpool(
        services.search_documents,
        resolved_root,
        query,
        mode,
        top_k_value,
    )
    summary = services.get_index_summary(resolved_root)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/search_results.html",
            {
                "request": request,
                "search": search_view,
            },
        )

    return templates.TemplateResponse(
        request,
        "index.html",
        _page_context(request, search=search_view, summary=summary),
    )


@app.post("/index", response_class=HTMLResponse)
async def rebuild_index(
    request: Request,
    root: str = Form(""),
) -> HTMLResponse:
    resolved_root = root or str(services.get_default_root())
    index_result = await run_in_threadpool(services.run_index, resolved_root)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/index_status.html",
            {
                "request": request,
                "index_result": index_result,
            },
        )

    search = services.search_documents(resolved_root, "", "hybrid", services.DEFAULT_TOP_K)
    return templates.TemplateResponse(
        request,
        "index.html",
        _page_context(
            request,
            search=search,
            summary=index_result.summary,
            index_result=index_result,
        ),
    )
