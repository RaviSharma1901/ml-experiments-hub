# Dockerfile

# base image
FROM python:3.12-slim

# work dircetory
WORKDIR /app

# copy
COPY requirements.txt ./

# run
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
