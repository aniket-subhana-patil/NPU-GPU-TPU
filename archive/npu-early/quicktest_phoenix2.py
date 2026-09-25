"""
quicktest_phoenix2.py -- Ryzen AI install test for Phoenix (7840HS),
robust to RYZEN_AI_INSTALLATION_PATH not being set.

Finds the install dir in this order:
  1. --install-dir argument if given
  2. RYZEN_AI_INSTALLATION_PATH env var if set
  3. default C:\\Program Files\\RyzenAI\\<newest version> if present

Also auto-locates the phoenix xclbin under the install dir instead of assuming
a fixed subpath (the voe-*/xclbins/phoenix location varies by version).

Usage (in the activated ryzen-ai conda env):
    python quicktest_phoenix2.py
    python quicktest_phoenix2.py --install-dir "C:\\Program Files\\RyzenAI\\1.7.1"
"""
import os, sys, subprocess, glob, argparse
import numpy as np
import onnxruntime as ort


def get_npu_info():
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    s = stdout.decode(errors='ignore')
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in s: return 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in s: return 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in s: return 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in s: return 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in s: return 'KRK'
    return ''


def find_install_dir(cli_arg):
    if cli_arg and os.path.isdir(cli_arg):
        return cli_arg
    env = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
    if env and os.path.isdir(env):
        return env
    # fall back to default location, pick newest version folder
    base = r'C:\Program Files\RyzenAI'
    if os.path.isdir(base):
        versions = sorted(glob.glob(os.path.join(base, '*')), reverse=True)
        for v in versions:
            if os.path.isdir(v):
                return v
    return None


def find_phoenix_xclbin(install_dir):
    # search anywhere under install for phoenix\*.xclbin (prefer 4x4)
    hits = glob.glob(os.path.join(install_dir, '**', 'xclbins', 'phoenix', '*.xclbin'),
                     recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(install_dir, '**', 'phoenix', '*.xclbin'),
                         recursive=True)
    # prefer 4x4 if present
    for h in hits:
        if '4x4' in os.path.basename(h):
            return h
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--install-dir', default=None)
    a = ap.parse_args()

    npu = get_npu_info()
    print(f'Detected NPU type: {npu or "UNKNOWN"}')

    install_dir = find_install_dir(a.install_dir)
    if not install_dir:
        sys.exit('Could not find Ryzen AI install dir. Pass --install-dir '
                 '"C:\\Program Files\\RyzenAI\\<version>"')
    print(f'Install dir: {install_dir}')

    model = os.path.join(install_dir, 'quicktest', 'test_model.onnx')
    if not os.path.isfile(model):
        # try to find test_model.onnx anywhere under install
        hits = glob.glob(os.path.join(install_dir, '**', 'test_model.onnx'), recursive=True)
        if hits:
            model = hits[0]
        else:
            sys.exit(f'test_model.onnx not found under {install_dir}')
    print(f'Model: {model}')

    providers = ['VitisAIExecutionProvider']
    provider_options = [{}]

    if npu == 'PHX/HPT':
        xclbin = find_phoenix_xclbin(install_dir)
        if not xclbin:
            sys.exit('Phoenix xclbin not found under install dir. '
                     'Run:  dir /s /b "%s\\*.xclbin"' % install_dir)
        print(f'Phoenix xclbin: {xclbin}')
        provider_options = [{
            'target': 'X1',
            'xlnx_enable_py3_round': 0,
            'xclbin': xclbin,
        }]

    so = ort.SessionOptions()
    so.log_severity_level = 0
    try:
        session = ort.InferenceSession(model, sess_options=so,
                                       providers=providers,
                                       provider_options=provider_options)
    except Exception as e:
        print(f'Failed to create InferenceSession: {e}')
        sys.exit(1)

    x = np.expand_dims(np.random.rand(3, 32, 32).astype(np.float32), axis=0)
    try:
        session.run(None, {'input': x})
    except Exception as e:
        print(f'Failed to run InferenceSession: {e}')
        sys.exit(1)
    print('Test finished')


if __name__ == '__main__':
    main()
