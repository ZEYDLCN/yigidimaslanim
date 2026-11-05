# Application image for the API and worker. Build context is prepared by
# scripts/build_wheelhouse.sh (offline install from the locked dependency set).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MIGRATIONS_DIR=/app/migrations

WORKDIR /app
COPY wheelhouse /wheels
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.lock.txt \
 && pip install --no-cache-dir --no-index --find-links /wheels --no-deps notes-graphrag \
 && rm -rf /wheels
COPY migrations ./migrations

RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
ENTRYPOINT ["notes-graphrag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
