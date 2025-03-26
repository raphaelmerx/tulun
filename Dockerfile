FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root && \
    pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu && \
    python -m spacy download en_core_web_sm

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8008

CMD ["daphne", "-b", "0.0.0.0", "-p", "8008", "tulun.asgi:application"]