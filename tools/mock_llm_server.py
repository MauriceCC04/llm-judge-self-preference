"""tools/mock_llm_server.py — tiny OpenAI-compatible mock server for local testing.

Starts an HTTP server on 127.0.0.1:<port> that returns stub JSON for any
chat/completions or responses call. Used by Gate-1 preflight and Gate-2 smoke
tests when a real vLLM is not available.

Usage::

    python tools/mock_llm_server.py --port 8765 &
    export TRAILTRAINING_LLM_BASE_URL=http://127.0.0.1:8765/v1
    export OPENROUTER_API_KEY=dummy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        from tests.mock_llm_client import _extract_schema_name, _build_payload

        schema_name = _extract_schema_name(body)
        payload = _build_payload(schema_name)
        text = json.dumps(payload, ensure_ascii=False)

        requested_model = str(body.get("model") or "mock-model").strip() or "mock-model"

        if self.path.endswith("/chat/completions"):
            response_obj = {
                "id": "mock-chatcmpl",
                "object": "chat.completion",
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        else:
            response_obj = {
                "id": "mock-response",
                "object": "response",
                "status": "completed",
                "model": requested_model,
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
            }

        wire = json.dumps(response_obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(wire)))
        self.end_headers()
        self.wfile.write(wire)

    def log_message(self, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"Mock LLM server running on http://127.0.0.1:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()