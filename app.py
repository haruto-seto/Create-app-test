"""Streamlit entrypoint for the LH2 dashboard."""

import importlib

import PH2_vital


# `from module import *` だとStreamlitのリロード時に再実行されず白画面になるため、
# 毎回モジュールをリロードして描画処理を確実に実行する。
importlib.reload(PH2_vital)