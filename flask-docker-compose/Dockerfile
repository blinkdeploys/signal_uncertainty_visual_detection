FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install matplotlib
COPY . .
CMD ["python", "app.py"]
