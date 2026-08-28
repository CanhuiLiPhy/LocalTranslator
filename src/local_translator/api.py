"""
中文：REST API 路由，供脚本调用与前端补充使用。

REST API surface.

The web UI drives everything through :class:`~local_translator.service.TranslatorService`
directly, so these endpoints exist for scripting and automation: submitting a
paper from a shell script, polling a job from another tool, or streaming
progress into an external dashboard.

They also serve the reader's PDF requests, which need path-based access to files
that live outside the library directory (translations written next to a source
PDF anywhere on disk).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SubmitPathRequest(BaseModel):
    """
    中文：按本地路径提交翻译的请求体。

    Request body for submitting a PDF already on disk.

    Attributes:
        path (str): Absolute or ``~``-relative path to the PDF.
        pages (str | None): BabelDOC page range expression, e.g. ``"1-5"``.
    """

    path: str
    pages: str | None = None


class SubmitFolderRequest(BaseModel):
    """
    中文：文件夹批量翻译的请求体。

    Request body for batch-importing a folder.

    Attributes:
        path (str): Directory to scan.
        recursive (bool): Descend into subdirectories.
        skip_translated (bool): Skip PDFs that already have translations.
        pages (str | None): Page range applied to every document.
    """

    path: str
    recursive: bool = True
    skip_translated: bool = True
    pages: str | None = None


class SubmitArxivRequest(BaseModel):
    """
    中文：arXiv 提交的请求体。

    Request body for fetching and translating an arXiv paper.

    Attributes:
        reference (str): arXiv id or URL.
        pages (str | None): Page range expression.
        as_copy (bool): Translate again as a numbered copy even when the
            library already holds a document with this title.
    """

    reference: str
    pages: str | None = None
    as_copy: bool = False


def register_api(app: FastAPI, service) -> None:
    """
    中文：把 REST 路由注册到 FastAPI 应用上。

    Attach the REST endpoints to the application.

    Args:
        app (FastAPI): The application instance (NiceGUI's underlying FastAPI app).
        service (TranslatorService): Service backing every endpoint.
    """

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        """
        中文：返回服务与本地模型的整体状态。

        Report overall service health.

        Returns:
            dict[str, Any]: Keys ``model`` (local server status), ``running``
                and ``queued`` job counts, and ``backend``.
        """
        snapshot = service.manager.snapshot()
        return {
            "model": service.model_status(),
            "running": len(snapshot["running"]),
            "queued": len(snapshot["queued"]),
            "backend": service.settings.translation.backend,
        }

    @app.get("/api/jobs")
    def list_jobs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        """
        中文：列出最近的任务。

        List recent jobs, newest first, with live progress merged in.

        Args:
            limit (int): Maximum number of jobs to return.

        Returns:
            list[dict[str, Any]]: Job records.
        """
        return service.dashboard()["recent"][:limit]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        """
        中文：查询单个任务。

        Fetch one job's full state.

        Args:
            job_id (str): Job identifier.

        Returns:
            dict[str, Any]: The job record.

        Raises:
            HTTPException: 404 if the job does not exist.
        """
        job = service.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        """
        中文：取消任务。

        Cancel a queued or running job.

        Args:
            job_id (str): Job identifier.

        Returns:
            dict[str, Any]: ``{"cancelled": bool}``.
        """
        return {"cancelled": await service.cancel(job_id)}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str) -> dict[str, Any]:
        """
        中文：重试失败或中断的任务。

        Re-enqueue a finished job's document.

        Args:
            job_id (str): Job identifier.

        Returns:
            dict[str, Any]: ``{"job_id": str}`` for the new job.

        Raises:
            HTTPException: 404 if the job or its source file is gone.
        """
        new_id = service.retry(job_id)
        if new_id is None:
            raise HTTPException(status_code=404, detail="Job or source file not found")
        return {"job_id": new_id}

    @app.post("/api/translate/path")
    async def submit_path(request: SubmitPathRequest) -> dict[str, Any]:
        """
        中文：提交本地 PDF 翻译。

        Translate a PDF already on disk, writing outputs beside it.

        Args:
            request (SubmitPathRequest): Path and options.

        Returns:
            dict[str, Any]: ``{"job_id": str}``.

        Raises:
            HTTPException: 404 if the file does not exist, 503 if the local
                model cannot be started.
        """
        source = Path(request.path).expanduser()
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {source}")
        await _ensure_model(service)
        return {"job_id": service.submit_path(source, pages=request.pages)}

    @app.post("/api/translate/folder")
    async def submit_folder(request: SubmitFolderRequest) -> dict[str, Any]:
        """
        中文：批量提交一个文件夹。

        Batch-translate every PDF in a folder.

        Args:
            request (SubmitFolderRequest): Folder and options.

        Returns:
            dict[str, Any]: Keys ``job_ids``, ``skipped``, ``total_found``.

        Raises:
            HTTPException: 404 if the folder does not exist.
        """
        folder = Path(request.path).expanduser()
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail=f"Not a directory: {folder}")
        await _ensure_model(service)
        return await asyncio.to_thread(
            service.submit_folder,
            folder,
            recursive=request.recursive,
            skip_translated=request.skip_translated,
            pages=request.pages,
        )

    @app.post("/api/translate/arxiv")
    async def submit_arxiv(request: SubmitArxivRequest) -> dict[str, Any]:
        """
        中文：下载并翻译一篇 arXiv 论文。

        Fetch an arXiv paper and translate it.

        Args:
            request (SubmitArxivRequest): Reference and options.

        Returns:
            dict[str, Any]: ``{"job_id": str}``.

        Raises:
            HTTPException: 400 if the reference cannot be resolved, 409 if the
                library already holds a paper with this title — resubmit with
                ``as_copy`` to translate it again as a numbered copy.
        """
        from local_translator.service import DuplicateDocument

        await _ensure_model(service)
        try:
            job_id = await service.submit_arxiv(
                request.reference, pages=request.pages, as_copy=request.as_copy)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DuplicateDocument as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "duplicate_title",
                    "title": exc.pending.title,
                    "copy_index": exc.copy_index,
                    "existing": [row.get("id") for row in exc.existing],
                    "hint": "resubmit with as_copy=true to translate a numbered copy",
                },
            ) from exc
        return {"job_id": job_id}

    @app.get("/api/queue")
    def queue() -> dict[str, Any]:
        """
        中文：返回真实的队列顺序（/api/jobs 是按创建时间倒序，不是队列顺序）。

        Report the scheduler's live queue in execution order.

        Distinct from ``/api/jobs``, which lists history newest-first: that
        ordering is the opposite of the queue's and using it to drive reordering
        moves the wrong job.

        Returns:
            dict[str, Any]: Keys ``running`` and ``queued``, the latter in the
                order jobs will actually start.
        """
        return service.manager.snapshot()

    @app.post("/api/jobs/{job_id}/move")
    def move_job(job_id: str, delta: int = Query(..., description="负数上移，正数下移")) -> dict[str, Any]:
        """
        中文：调整排队任务的顺序。

        Reorder a queued job.

        Args:
            job_id (str): Job to move.
            delta (int): Negative moves it earlier, positive later.

        Returns:
            dict[str, Any]: ``{"moved": bool}``.
        """
        return {"moved": service.manager.reorder(job_id, delta)}

    @app.post("/api/queue/clear")
    async def clear_queue() -> dict[str, Any]:
        """
        中文：清空队列（不影响正在运行的任务）。

        Cancel all queued jobs.

        Returns:
            dict[str, Any]: ``{"cleared": int}``.
        """
        return {"cleared": await service.manager.clear_queue()}

    @app.get("/api/documents")
    def list_documents(limit: int = Query(200, ge=1, le=1000)) -> list[dict[str, Any]]:
        """
        中文：列出文库中的文档。

        List library documents.

        Args:
            limit (int): Maximum rows to return.

        Returns:
            list[dict[str, Any]]: Document records with their latest job summary.
        """
        return service.documents(limit=limit)

    @app.get("/api/usage")
    def usage() -> dict[str, Any]:
        """
        中文：返回累计 token 用量与费用（按后端分组）。

        Aggregate token usage across all completed jobs.

        Returns:
            dict[str, Any]: Backend name to usage totals.
        """
        return service.db.usage_totals()

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        """
        中文：以 SSE 形式推送任务事件流。

        Stream job events as Server-Sent Events.

        Provided for external tools; the web UI polls instead, which keeps a
        refreshed page from having to re-establish and reconcile a subscription.

        Returns:
            StreamingResponse: An ``text/event-stream`` response.
        """

        async def generate():
            async for event in service.manager.subscribe():
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/pdf/{doc_id}/{kind}")
    def serve_pdf(doc_id: str, kind: str) -> FileResponse:
        """
        中文：按文档 id 与类型提供 PDF 给阅读器。

        Serve one of a document's PDFs to the reader.

        Addressing by ``(doc_id, kind)`` rather than by filesystem path solves
        two problems at once. It removes the arbitrary-path parameter, so this
        endpoint cannot be turned into a file-disclosure hole regardless of what
        a caller supplies. And it keeps the URL free of percent-encoded path
        separators, which matters because the URL is embedded as pdf.js's
        ``file=`` query parameter — an absolute path encoded there survives one
        round of encoding too few or too many and fetches as a 404.

        Args:
            doc_id (str): Document identifier.
            kind (str): ``source``, ``dual`` or ``mono``.

        Returns:
            FileResponse: The PDF, served with range support so pdf.js can fetch
                it page by page.

        Raises:
            HTTPException: 400 for an unknown kind, 404 if the document, the
                translation, or the file on disk is missing.
        """
        if kind not in ("source", "dual", "mono"):
            raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}")

        document = service.db.get_document(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail=f"No such document: {doc_id}")

        if kind == "source":
            candidate = document["source_path"]
        else:
            job = service.db.query_one(
                f"""
                SELECT {kind}_path AS p FROM jobs
                WHERE doc_id = ? AND status = 'completed' AND {kind}_path IS NOT NULL
                ORDER BY finished_at DESC LIMIT 1
                """,
                (doc_id,),
            )
            candidate = job["p"] if job else None

        if not candidate:
            raise HTTPException(status_code=404, detail=f"No {kind} PDF for this document")

        file_path = Path(candidate)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File no longer exists on disk")

        return FileResponse(file_path, media_type="application/pdf", filename=file_path.name)


async def _ensure_model(service) -> None:
    """
    中文：确保本地模型就绪，失败时返回 503。

    Ensure the local inference server is reachable before queueing work.

    Args:
        service (TranslatorService): Service owning the supervisor.

    Raises:
        HTTPException: 503 if the local model cannot be started.
    """
    from local_translator.service import LlamaServerError

    try:
        await asyncio.to_thread(service.ensure_model_ready)
    except LlamaServerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
