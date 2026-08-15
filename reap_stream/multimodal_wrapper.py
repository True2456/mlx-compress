"""OpenAI-compatible proxy that gives Ling-3.0-flash vision/audio capability
by relaying image/audio content to Gemma-4 for description, then feeding
the result to Ling as text -- without any model surgery or training.

Not true joint multimodal understanding (Ling never sees pixels/waveforms
directly, only Gemma's text description of them), but it's the practical,
zero-training answer: one endpoint, one model to point a client at, both
capabilities. Plain text-only requests pass straight through to Ling with
no extra hop.

Talks to oMLX's own already-running OpenAI-compatible API
(localhost:<port from ~/.omlx/settings.json>) for both the Gemma-4 caption
step and the final Ling call -- no new model loading, oMLX already hosts
both models simultaneously.

Usage:
    .venv/bin/python -m reap_stream.multimodal_wrapper --port 8100
    # then point any OpenAI-compatible client at http://localhost:8100/v1
    # with model="ling-multimodal" (or any name -- it's remapped internally)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("multimodal_wrapper")
logging.basicConfig(level=logging.INFO)


def _load_omlx_settings() -> dict[str, Any]:
    path = Path.home() / ".omlx" / "settings.json"
    return json.loads(path.read_text())


def _omlx_base_url() -> str:
    settings = _load_omlx_settings()
    port = settings.get("server", {}).get("port", 1235)
    return f"http://127.0.0.1:{port}"


def _omlx_api_key() -> str | None:
    settings = _load_omlx_settings()
    return settings.get("auth", {}).get("api_key")


def _has_multimodal_content(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_audio"):
                    return True
    return False


def _extract_multimodal_parts(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split each message's content into (text_parts, mm_parts), per message index."""
    per_message = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            per_message.append(([], []))
            continue
        text_parts = [p for p in content if p.get("type") == "text"]
        mm_parts = [p for p in content if p.get("type") in ("image_url", "input_audio")]
        per_message.append((text_parts, mm_parts))
    return per_message


def build_app(caption_model: str, target_model: str, timeout: float = 120.0) -> FastAPI:
    app = FastAPI(title="ling-multimodal-wrapper")
    base_url = _omlx_base_url()
    api_key = _omlx_api_key()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def _caption_multimodal_parts(mm_parts: list[dict]) -> str:
        """Send image/audio parts to Gemma-4, get back a text description."""
        instruction = (
            "Describe this content in detail and objectively -- if it's an "
            "image, describe what's shown; if it's audio, transcribe/summarize "
            "it. Be thorough, since your description is the only information "
            "another model will have about this content."
        )
        content = [{"type": "text", "text": instruction}] + mm_parts
        payload = {
            "model": caption_model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/v1/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _rewrite_messages(messages: list[dict]) -> list[dict]:
        """Replace image_url/input_audio parts with Gemma's text description."""
        rewritten = []
        for msg, (text_parts, mm_parts) in zip(messages, _extract_multimodal_parts(messages)):
            if not mm_parts:
                rewritten.append(msg)
                continue
            logger.info("Captioning %d multimodal part(s) via %s", len(mm_parts), caption_model)
            description = await _caption_multimodal_parts(mm_parts)
            new_content = list(text_parts) + [
                {"type": "text", "text": f"[Attached content, described by {caption_model}]:\n{description}"}
            ]
            new_msg = dict(msg)
            new_msg["content"] = new_content
            rewritten.append(new_msg)
        return rewritten

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])

        if _has_multimodal_content(messages):
            body = dict(body)
            body["messages"] = await _rewrite_messages(messages)
            logger.info("Rewrote multimodal request -> text-only, forwarding to %s", target_model)
        else:
            logger.debug("Plain text request, passing through to %s untouched", target_model)

        body["model"] = target_model
        stream = bool(body.get("stream"))

        if not stream:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{base_url}/v1/chat/completions", json=body, headers=headers
                )
                return JSONResponse(status_code=resp.status_code, content=resp.json())

        async def _stream():
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{base_url}/v1/chat/completions", json=body, headers=headers
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": target_model, "object": "model", "owned_by": "ling-multimodal-wrapper"}],
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "caption_model": caption_model, "target_model": target_model}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--caption-model", default="gemma-4-12b-it-qat-4bit-frontierdistill")
    ap.add_argument("--target-model", default="Ling-3.0-flash-8fixed-4routed")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    import uvicorn
    app = build_app(args.caption_model, args.target_model, args.timeout)
    logger.info(
        "Starting multimodal wrapper on :%d -> caption=%s, target=%s, omlx=%s",
        args.port, args.caption_model, args.target_model, _omlx_base_url(),
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
