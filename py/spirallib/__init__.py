
from datetime import datetime
import humanize
import os.path as osp
import time
import pyttsx3

t0 = t1 = time.time()
speech_engine = pyttsx3.init()

# Get the Storage object
from .notebook_utils import NotebookUtilities
nu = NotebookUtilities(
    data_folder_path=osp.abspath('../data'),
    saves_folder_path=osp.abspath('../saves')
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