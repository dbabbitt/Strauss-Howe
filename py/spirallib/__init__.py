
from datetime import datetime
import humanize
import time
import pyttsx3

t0 = t1 = time.time()
speech_engine = pyttsx3.init()

# Add the path to the shared utilities directory
import os.path as osp

# Define the shared folder path using join for better compatibility
shared_folder = osp.abspath(osp.join(
    osp.dirname(__file__), '..', '..', '..', 'share'
))

# Add the shared folder to sys.path if it's not already included
import sys
if shared_folder not in sys.path:
    sys.path.insert(1, shared_folder)

# Attempt to import the Storage object
try:
    from notebook_utils import NotebookUtilities
except ImportError as e:
    print(f"Error importing NotebookUtilities: {e}")

# Initialize with data and saves folder paths
nu = NotebookUtilities(
    data_folder_path=osp.abspath(osp.join(
        osp.dirname(__file__), '..', '..', 'data'
    )),
    saves_folder_path=osp.abspath(osp.join(
        osp.dirname(__file__), '..', '..', 'saves'
    ))
)
secrets_json_path = osp.abspath(osp.join(nu.data_folder, 'json', 'secrets.json'))

# Get the StraussHoweUtilities object
from .spiral_utils import StraussHoweUtilities
shu = StraussHoweUtilities(
    s=nu
)

duration_str = humanize.precisedelta(time.time() - t1, minimum_unit='seconds', format='%0.0f')
speech_str = f'Utility libraries created in {duration_str}'
print(speech_str)
speech_engine.say(speech_str)
speech_engine.runAndWait()

print(f"from spirallib import ({', '.join(dir())})")
print(r'\b(' + '|'.join(dir()) + r')\b')