import subprocess
import sys

cmd = [
    sys.executable,
    "/project/models/check_fraud_alerts.py"
]

subprocess.run(cmd, check=True)
