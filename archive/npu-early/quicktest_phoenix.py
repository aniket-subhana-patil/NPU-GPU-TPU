"""
quicktest_phoenix.py -- Ryzen AI installation test, PATCHED FOR PHOENIX (7840HS).

The stock quicktest.py in the Ryzen AI install assumes Strix or newer. This
version (from the AMD docs) auto-detects Phoenix/Hawk Point and sets the required
provider options: target 'X1' + the phoenix xclbin. Run this instead of the
stock test to verify your 7840HS NPU actually runs a model.

Usage (in the Miniforge Prompt, with the ryzen-ai conda env activated):
    copy this over %RYZEN_AI_INSTALLATION_PATH%\\quicktest\\quicktest.py
    (or run it from that folder so it finds test_model.onnx)

    python quicktest_phoenix.py
    # verbose node-placement check:
    python quicktest_phoenix.py 2>&1 | findstr /i "VerifyEachNodeIsAssignedToAnEp Test"
"""
import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort


def get_npu_info():
    # Enumerate PCI devices to identify the NPU generation.
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    npu_type = ''
    s = stdout.decode(errors='ignore')
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in s: npu_type = 'PHX/HPT'   # <-- your 7840HS
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in s: npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in s: npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in s: npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in s: npu_type = 'KRK'
    return npu_type


npu_type = get_npu_info()
print(f'Detected NPU type: {npu_type or "UNKNOWN"}')

install_dir = os.environ['C:/minif/envs']
model = os.path.join(install_dir, 'quicktest', 'test_model.onnx')
providers = ['VitisAIExecutionProvider']
provider_options = [{}]   # default (STX/KRK and newer)

if npu_type == 'PHX/HPT':
    print('Setting environment for PHX/HPT (Phoenix) -> target X1 + phoenix xclbin')
    xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
    provider_options = [{
        'target': 'X1',
        'xlnx_enable_py3_round': 0,
        'xclbin': xclbin_file,
    }]

session_options = ort.SessionOptions()
session_options.log_severity_level = 0   # 0=Verbose so we can see node placement

try:
    session = ort.InferenceSession(model,
                                   sess_options=session_options,
                                   providers=providers,
                                   provider_options=provider_options)
except Exception as e:
    print(f'Failed to create an InferenceSession: {e}')
    sys.exit(1)

def preprocess_random_image():
    image_array = np.random.rand(3, 32, 32).astype(np.float32)
    return np.expand_dims(image_array, axis=0)

input_data = preprocess_random_image()
try:
    outputs = session.run(None, {'input': input_data})
except Exception as e:
    print(f'Failed to run the InferenceSession: {e}')
    sys.exit(1)
else:
    print('Test finished')
