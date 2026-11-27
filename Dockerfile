FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /audit
COPY . /audit
RUN pip install --no-cache-dir torch==2.0.1 numpy==1.23.5 opencv-python-headless==4.11.0.86 \
    mmengine==0.8.3 mmpretrain==1.0.0 pillow==10.4.0 matplotlib==3.8.4 pyyaml==6.0.2
CMD ["python", "audit/unit_test.py"]
