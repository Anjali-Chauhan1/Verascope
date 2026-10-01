"""
Hugging Face Spaces Entry Point
Launches the Verascope Streamlit interface from the root directory.
"""
import os
import sys

# Ensure current directory is in Python path
sys.path.insert(0, os.path.abspath("."))

# Run the primary Streamlit app
from app.streamlit_app import *
