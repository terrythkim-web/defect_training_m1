import os

# Headless Qt for CI (Linux and Windows) and local test runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
