FROM python:3.12-slim

WORKDIR /app

# Install lightweight backend dependencies
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.15.0" "torchvision==0.29.2" \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend/ ./backend/
COPY ml/ ./ml/

# Copy trained checkpoint
COPY artifacts/models/mobilenetv3_best.pth ./artifacts/models/mobilenetv3_best.pth

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]