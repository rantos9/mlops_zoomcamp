FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "predict_script.py", "./"]

RUN pip install -U pip && \
    pip install pipenv && \
    pipenv install --system --deploy

CMD ["python", "predict_script.py"]