"""
Entry point for Hugging Face Spaces' Gradio SDK.

The Space's account can't use the Docker SDK (needs verification/Pro), so
this runs on the Gradio SDK instead — which is really just "a free Python
environment that runs app.py and expects a server on port 7860." Gradio is
built on FastAPI, so the real app (server.py's FastAPI instance, with all of
The Box's routes already on it) gets mounted as the actual thing users hit;
the Gradio Blocks object below is a required-but-unused placeholder, pushed
to /gradio where nobody will look. No Gradio UI code is otherwise involved —
the room, the transcript, the sound, all of it is server.py + web/index.html
exactly as they run locally.

The account's free hardware tier is ZeroGPU, not CPU Basic (downgrading
needs Pro). ZeroGPU refuses to start any app that has zero @spaces.GPU
functions, even though this app does no local GPU work at all — every
"thinking" step is a remote call to the Gemini API. _keepalive below exists
purely to satisfy that startup check; it is never called.
"""

import gradio as gr

try:
    import spaces

    @spaces.GPU
    def _keepalive():
        return None
except ImportError:
    # Not running on a ZeroGPU Space (e.g. local dev) — nothing to satisfy.
    pass

from server import app as fastapi_app

_placeholder = gr.Blocks()
with _placeholder:
    gr.Markdown("The Box runs at the root URL, not here.")

app = gr.mount_gradio_app(fastapi_app, _placeholder, path="/gradio")

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
