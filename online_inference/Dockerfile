FROM python:slim
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY models/model.pkl /model.pkl
COPY models/transformer.pkl /transformer.pkl
COPY app.py /app.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"
ENV PATH_TO_TRANSFORMER="/transformer.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]