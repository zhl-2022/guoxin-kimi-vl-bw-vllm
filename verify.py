"""Current full OCRBench evaluator; older publisher011 failures remain archived."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent/'scripts/verify_kimi_ocr016.py'),run_name='__main__')
