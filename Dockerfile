FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv
ARG PIP_NO_INDEX
ARG PIP_FIND_LINKS
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --no-compile -r requirements.txt
COPY . .
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=12s --start-period=90s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).close()"]
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16"]
