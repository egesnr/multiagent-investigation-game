"""
Kept so an existing host whose start command names app.py keeps working:
`python app.py` and `uvicorn app:app` both run server.py's FastAPI app.
"""

import os

from server import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
