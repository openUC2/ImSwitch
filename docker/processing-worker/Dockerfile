# Processing worker for ImSwitch: runs the deconvolution jobs a FRAME sends (docs/DECONVOLUTION.md).
# The build context is the folder that holds the recipe, so the image carries two files:
#
#   docker build -f docker/processing-worker/Dockerfile -t imswitch-processing-worker imswitch/imcontrol/model/aufgaben
#   docker run --rm -p 8090:8090 imswitch-processing-worker
#
# Then in ImSwitch: OME-Zarr Viewer > Deconvolve > Worker URL http://<this computer>:8090.
# The worker never connects to the microscope; it only has to be reachable from it.
FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy>=1.26" "scipy>=1.11" \
    && useradd --create-home worker
WORKDIR /app
COPY deconvolution.py worker.py ./
USER worker
EXPOSE 8090
CMD ["python", "-u", "worker.py", "--port", "8090"]
