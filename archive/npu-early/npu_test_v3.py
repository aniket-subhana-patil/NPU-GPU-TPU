"""
npu_test_v3.py -- Phoenix NPU test using the DOCUMENTED runtime_setup approach:
sets XLNX_VART_FIRMWARE + XLNX_TARGET_NAME to the phoenix 1x4.xclbin (per AMD
docs runtime_setup page), instead of passing xclbin via provider_options.

This differs from earlier attempts: uses 1x4.xclbin (not 4x4) and the firmware
env-var mechanism the docs specify for PHX/HPT.

Run in the activated ryzen-ai conda env:
    python npu_test_v3.py
    python npu_test_v3.py --install-dir "C:\\Program Files\\RyzenAI\\1.7.1"
"""
import os, sys, subprocess, glob, argparse
import numpy as np


def get_npu():
    p = subprocess.Popen(r'pnputil /enum-devices /bus PCI /deviceids ',
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s = p.communicate()[0].decode(errors='ignore')
    if 'VEN_1022&DEV_1502&REV_00' in s: return 'PHX/HPT'
    if 'VEN_1022&DEV_17F0' in s: return 'STX'
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--install-dir', default=None)
    a = ap.parse_args()

    npu = get_npu()
    print(f'NPU type: {npu or "UNKNOWN"}')

    install = a.install_dir or os.environ.get('RYZEN_AI_INSTALLATION_PATH')
    if not install or not os.path.isdir(install):
        base = r'C:\Program Files\RyzenAI'
        vers = sorted(glob.glob(os.path.join(base, '*')), reverse=True)
        install = next((v for v in vers if os.path.isdir(v)), None)
    if not install:
        sys.exit('install dir not found; pass --install-dir')
    print(f'Install: {install}')

    # DOCUMENTED Phoenix firmware config (runtime_setup page):
    #   XLNX_VART_FIRMWARE -> phoenix/1x4.xclbin
    #   XLNX_TARGET_NAME   -> AMD_AIE2_Nx4_Overlay
    if npu == 'PHX/HPT':
        # locate the 1x4 phoenix xclbin (search if the default path differs)
        cand = os.path.join(install, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '1x4.xclbin')
        if not os.path.isfile(cand):
            hits = glob.glob(os.path.join(install, '**', 'phoenix', '1x4.xclbin'), recursive=True)
            cand = hits[0] if hits else None
        if not cand:
            # fall back to any phoenix xclbin
            hits = glob.glob(os.path.join(install, '**', 'phoenix', '*.xclbin'), recursive=True)
            cand = hits[0] if hits else None
        if not cand:
            sys.exit('No phoenix xclbin found under install dir.')
        os.environ['XLNX_VART_FIRMWARE'] = cand
        os.environ['XLNX_TARGET_NAME'] = 'AMD_AIE2_Nx4_Overlay'
        print(f'XLNX_VART_FIRMWARE = {cand}')
        print(f'XLNX_TARGET_NAME   = AMD_AIE2_Nx4_Overlay')

    # import ORT only AFTER env vars are set
    import onnxruntime as ort

    model = os.path.join(install, 'quicktest', 'test_model.onnx')
    if not os.path.isfile(model):
        hits = glob.glob(os.path.join(install, '**', 'test_model.onnx'), recursive=True)
        model = hits[0] if hits else None
    if not model:
        sys.exit('test_model.onnx not found')
    print(f'Model: {model}')

    so = ort.SessionOptions()
    so.log_severity_level = 1

    # With the firmware env-var approach, provider_options can just set target.
    provider_options = [{'target': 'X1'}] if npu == 'PHX/HPT' else [{}]

    try:
        sess = ort.InferenceSession(model, sess_options=so,
                                    providers=['VitisAIExecutionProvider'],
                                    provider_options=provider_options)
    except Exception as e:
        print(f'Session creation FAILED: {e}')
        sys.exit(1)

    x = np.expand_dims(np.random.rand(3, 32, 32).astype(np.float32), 0)
    try:
        sess.run(None, {'input': x})
    except Exception as e:
        print(f'Inference FAILED: {e}')
        sys.exit(1)
    print('SUCCESS: model ran on NPU')


if __name__ == '__main__':
    main()
