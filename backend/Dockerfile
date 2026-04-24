# Hugging Face Spaces requires:
#   - port 7860
#   - non-root user (uid 1000)


FROM python:3.10-slim

RUN useradd -m -u 1000 user && mkdir -p /app && chown user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies app.py, mobilenet_best.keras, and anything else in backend/
COPY --chown=user . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
