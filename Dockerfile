# SmartBeam 2.0 container (used by the Hugging Face Space; works on any Docker host).
FROM python:3.12-slim

# Hugging Face Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Do slow one-time work at build time, not on the first visit: Matplotlib font cache + bytecode.
RUN python -c "import matplotlib.pyplot" && python -m compileall -q smartbeam views app.py serve.py

# Render/Railway/etc. inject $PORT; Hugging Face uses 7860.
ENV PORT=7860
EXPOSE 7860
# XSRF off: hosts that embed the app in an iframe (e.g. Hugging Face) break Streamlit's XSRF cookie on uploads.
CMD ["sh", "-c", "python serve.py --server.port=${PORT} --server.address=0.0.0.0 --server.enableXsrfProtection=false"]
