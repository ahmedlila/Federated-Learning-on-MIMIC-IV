FROM python:3.9
WORKDIR /app
COPY . /app
COPY ../shared /app/shared
RUN pip install torch requests
CMD ["python", "node.py"]