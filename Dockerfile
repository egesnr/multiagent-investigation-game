FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py models.py agents.py game_logic.py resolution.py llm_utils.py server.py ./
COPY case_files ./case_files
COPY web ./web

# Hugging Face Spaces routes traffic to this port by convention.
EXPOSE 7860
ENV PORT=7860

CMD ["python", "server.py"]
