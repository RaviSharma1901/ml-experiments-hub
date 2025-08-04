FROM python:3.12-slim

WORKDIR /app

# Copy requirement file separately first
COPY requirements.txt .

# Install dependencies (includes streamlit)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code AFTER installing dependencies
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
