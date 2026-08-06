FROM python@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
WORKDIR /repro
COPY . /repro
USER 65534:65534
CMD ["python3", "/repro/reproduce.py"]
