FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Pinned dependencies first, so code changes do not reinstall them.
COPY requirements.txt ./
RUN grep -v '^-e ' requirements.txt > /tmp/requirements.txt && pip install -r /tmp/requirements.txt

# Editable install: config.py locates models/ and reports/ relative to the source tree.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

COPY config ./config
COPY models ./models
COPY reports/metrics.json ./reports/metrics.json
COPY streamlit_app.py ./

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
