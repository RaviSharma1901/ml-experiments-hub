FROM python:3.12-slim
WORKDIR /fraud_feeder
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . . 
CMD ["python", "feeder_batch.py"]
