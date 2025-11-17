import json
import logging
from datetime import datetime
from django.conf import settings
import os

# Ensure a folder for logs exists
log_dir = os.path.join(settings.BASE_DIR, ".logs")
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file = os.path.join(log_dir, "user_behavior.log")

# Set up logger
logger = logging.getLogger("user_behavior")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_user_action(session_key, action, details=None):
    """
    Log structured user behavior.

    Args:
        session_key (str): The user's session key.
        action (str): Name of the action, e.g. 'upload_document'
        details (dict, optional): Extra data about the action
    """
    log_entry = {
        "session_key": session_key,
        "action": action,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    logger.info(json.dumps(log_entry))
