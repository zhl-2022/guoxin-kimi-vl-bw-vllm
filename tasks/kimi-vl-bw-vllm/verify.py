#!/usr/bin/env python3
"""Transparent canonical entry; no fabricated official evaluation."""
import os
from pathlib import Path
import sys
entry=Path(__file__).resolve().with_name("verify.sh")
os.execvp("bash",["bash",str(entry),*sys.argv[1:]])
