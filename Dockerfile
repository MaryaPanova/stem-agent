# A self-modifying agent with web access and code execution must be contained.
# This image is that container: the agent runs its rollouts, web fetches, and
# run_python tool calls inside here, not on the host.
FROM python:3.12-slim

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir \
    "pydantic>=2" "rich>=13" "typer>=0.12" "python-dotenv>=1" "httpx>=0.27" "anthropic>=0.40" "pytest>=8"

COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Default: the tokenless smoke test, so `docker run` proves the pipeline with no key.
# Override the command to run the real eval, e.g.:
#   docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY stem-agent \
#       python main.py eval --domain trading -g 3
CMD ["python", "main.py", "eval", "--mock"]
