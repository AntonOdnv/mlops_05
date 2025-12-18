FROM --platform=linux/amd64 python:3.10.12

WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app.py .
COPY model.joblib .
COPY logger_cfg.yaml .

CMD ["python", "app.py"]
