#!/usr/bin/env python3
"""
npu_stage2_quantize_parallel.py -- Quantize all 1000 NAS-Bench ONNX models to
int8 for the Ryzen AI NPU using AMD Quark, PARALLELIZED across a few workers.

Same per-model config as the single-model version (confirmed working):
PowerOfTwoMethod.MinMSE, QUInt8 activations / QInt8 weights, NPU CNN mode,
symmetric activations, 100 calibration samples.

Quark uses ~0.74GB per instance, so 2-3 workers is safe on 32GB RAM.
Default is 2 workers to be safe; bump with --workers 3 if RAM allows.

Run INSIDE the activated ryzen-ai-1.7.1 conda env.

    conda activate ryzen-ai-1.7.1
    python npu_stage2_quantize_parallel.py --onnx-dir npu_study\\onnx --out-dir npu_study\\onnx_int8 --workers 2

Resumable: skips models whose int8 .onnx already exists (so the 1 you already
did won't be re-run).
"""
import argparse, glob, os, sys, time
import multiprocessing as mp
import numpy as np


class RandomCalibrationDataReader:
    def __init__(self, input_name, n_samples=100, shape=(1, 32, 32, 3)):
        self.input_name = input_name
        self.n = n_samples
        self.shape = shape
        self._i = 0
    def get_next(self):
        if self._i >= self.n:
            return None
        self._i += 1
        return {self.input_name: np.random.rand(*self.shape).astype(np.float32)}
    def rewind(self):
        self._i = 0


def _worker_init():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore')


def _quantize_one(args):
    mp_path, out_dir, calib_samples = args
    h = os.path.splitext(os.path.basename(mp_path))[0]
    out_path = os.path.join(out_dir, f'{h}.int8.onnx')
    if os.path.exists(out_path):
        return (h, 'skip')
    try:
        import onnx
        from quark.onnx import ModelQuantizer
        from quark.onnx.quantization.config.config import Config, QuantizationConfig
        from onnxruntime.quantization import QuantType
        try:
            from quark.onnx import PowerOfTwoMethod
        except Exception:
            from onnxruntime.quantization.calibrate import CalibrationMethod as PowerOfTwoMethod

        m = onnx.load(mp_path)
        input_name = m.graph.input[0].name
        dr = RandomCalibrationDataReader(input_name, n_samples=calib_samples)

        quant_config = QuantizationConfig(
            calibrate_method=PowerOfTwoMethod.MinMSE,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            enable_npu_cnn=True,
            extra_options={'ActivationSymmetric': True},
        )
        config = Config(global_quant_config=quant_config)
        quantizer = ModelQuantizer(config)
        quantizer.quantize_model(mp_path, out_path, dr)
        return (h, 'ok')
    except Exception as e:
        return (h, f'error:{type(e).__name__}:{str(e)[:80]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx-dir', required=True)
    ap.add_argument('--out-dir', default='npu_study/onnx_int8')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--calib-samples', type=int, default=100)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    models = sorted(glob.glob(os.path.join(a.onnx_dir, '*.onnx')))
    models = [m for m in models if not m.endswith('.int8.onnx')]
    todo = [m for m in models
            if not os.path.exists(os.path.join(
                a.out_dir, os.path.splitext(os.path.basename(m))[0] + '.int8.onnx'))]
    print(f'{len(models)} total, {len(models)-len(todo)} already done, '
          f'{len(todo)} to quantize across {a.workers} workers.\n')

    if not todo:
        print('All done already.')
        return

    tasks = [(m, a.out_dir, a.calib_samples) for m in todo]
    t0 = time.time()
    ok = err = 0
    with mp.Pool(a.workers, initializer=_worker_init) as pool:
        for i, (h, status) in enumerate(pool.imap_unordered(_quantize_one, tasks), 1):
            if status in ('ok', 'skip'):
                ok += 1
            else:
                err += 1
                print(f'  {h[:10]} {status}')
            if i % 10 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f'  [{i}/{len(todo)}] ok={ok} err={err} | '
                      f'{rate*60:.1f}/min | ETA {eta:.0f} min')

    dt = (time.time() - t0) / 60
    print(f'\nDone in {dt:.1f} min. {ok} int8 models in {a.out_dir} (errors: {err})')
    print('Next: time these on the NPU.')


if __name__ == '__main__':
    main()
