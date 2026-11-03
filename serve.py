"""Container entry point: import the heavy numeric stack once at boot, then start Streamlit in this process.

Streamlit runs page scripts in the server process, so modules imported here are already in sys.modules when a
visitor opens a page. On a small shared CPU this moves ~3 s x CPU-slowdown of imports from the first click to boot.
"""
import sys

import matplotlib.figure  # noqa: F401
import numpy  # noqa: F401
import pandas  # noqa: F401
import plotly.graph_objects  # noqa: F401
import reportlab.platypus  # noqa: F401
import scipy.linalg  # noqa: F401
import scipy.optimize  # noqa: F401
import scipy.stats  # noqa: F401

import smartbeam.analytical  # noqa: F401
import smartbeam.fem  # noqa: F401
import smartbeam.lab  # noqa: F401
import smartbeam.optimize  # noqa: F401
import smartbeam.report  # noqa: F401
import smartbeam.viz3d  # noqa: F401
from streamlit.web.cli import main

sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
sys.exit(main())
