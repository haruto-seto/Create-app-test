"""Backward-compatible Streamlit entrypoint.

Some environments still run `streamlit run app.py`.
This file keeps that command working while the main app lives in PH2_vital.py.
"""

# Keep this file as a minimal compatibility shim.
# The actual Streamlit app is defined in PH2_vital.py.
import PH2_vital  # noqa: F401
