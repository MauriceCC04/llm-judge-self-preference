"""tools/mock_llm_server.py — tiny OpenAI-compatible mock server for local testing.

Starts an HTTP server on 127.0.0.1:<port> that returns stub JSON for any
chat/completions or responses call.  Used by Gate-1 preflight and Gate-2 smoke
tests when a real vLLM is not available.

Usage::

    python tools/mock_llm_server.py --port 8765 &
    export TRAILTRAINING_LLM_BASE_URL=http://127.0.0.1:8765/v1
    export OPENROUTER_API_KEY=dummy
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        # Health endpoint
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        # Detect schema from body if available
        from tests.mock_llm_client import _extract_schema_name, _build_payload
        schema_name = _extract_schema_name(body)
        payload = _build_payload(schema_name)
        text = json.dumps(payload, ensure_ascii=False)

        response = json.dumps({
            "id": "mock-response",
            "object": "response",
            "output_text": text,
            "output": [{"type": "text", "text": text}],
            "choices": [{"message": {"content": text, "role": "assistant"}}],
        })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, *args: object) -> None:
        pass  # silence access log


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
