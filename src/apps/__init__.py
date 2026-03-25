"""应用入口模块"""

import sys
from pathlib import Path

# 确保 src 目录在 Python 路径中
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
