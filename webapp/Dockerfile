FROM python:3.12-slim
WORKDIR /app
COPY app.py .
RUN pip install -U pip && \
    pip install -U chainlit && \
    pip install -U numpy && \
    pip install -U pillow
EXPOSE 8080
CMD ["/bin/bash", "-c", "chainlit run app.py -h --host 0.0.0.0 --port 8080"]

