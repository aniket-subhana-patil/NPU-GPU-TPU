#!/usr/bin/env python3
"""
npu_stage2_quantize.py -- Quantize NAS-Bench ONNX models to int8 for the Ryzen AI
NPU, using AMD Quark (the quantizer shipped with Ryzen AI 1.7.x).

Uses the AMD-recommended CNN-for-NPU config: PowerOfTwoMethod.MinMSE calibration,
QUInt8 activations / QInt8 weights, NPU CNN mode, symmetric activations. Random
calibration data (32x32x3, CIFAR-like) is sufficient for latency work.

Run INSIDE the activated ryzen-ai-1.7.1 conda env (that's where quark lives).

    conda activate ryzen-ai-1.7.1

    # test ONE model first (recommended, to check the missing-cl-compiler warning
    # doesn't block quantization):
    python npu_stage2_quantize.py --onnx-dir npu_study\\onnx --out-dir npu_study\\onnx_int8 --limit 1

    # then all of them:
    python npu_stage2_quantize.py --onnx-dir npu_study\\onnx --out-dir npu_study\\onnx_int8

Resumable: skips models whose int8 .onnx already exists.
"""
import argparse, glob, os, sys
import numpy as np


class RandomCalibrationDataReader:
    """Feeds random 32x32x3 samples as calibration data (CIFAR-shaped)."""
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


def get_input_name(onnx_path):
    import onnx
    m = onnx.load(onnx_path)
    return m.graph.input[0].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx-dir', required=True, help='folder of float32 .onnx models')
    ap.add_argument('--out-dir', default='npu_study/onnx_int8')
    ap.add_argument('--limit', type=int, default=None, help='quantize only N (for testing)')
    ap.add_argument('--calib-samples', type=int, default=100)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    # Import Quark (confirmed available API surface for 1.7.x)
    try:
        from quark.onnx import ModelQuantizer
        from quark.onnx.quantization.config.config import Config, QuantizationConfig
        from onnxruntime.quantization import QuantType
        # calibrate method enum location can vary; import defensively
        try:
            from quark.onnx import PowerOfTwoMethod
        except Exception:
            from onnxruntime.quantization.calibrate import CalibrationMethod as PowerOfTwoMethod  # fallback
    except ImportError as e:
        sys.exit(f'Quark import failed: {e}\nRun inside the ryzen-ai conda env.')

    models = sorted(glob.glob(os.path.join(a.onnx_dir, '*.onnx')))
    models = [m for m in models if not m.endswith('.int8.onnx')]
    if a.limit:
        models = models[:a.limit]
    print(f'{len(models)} models to quantize.\n')

    ok = err = 0
    for i, mp in enumerate(models, 1):
        h = os.path.splitext(os.path.basename(mp))[0]
        out_path = os.path.join(a.out_dir, f'{h}.int8.onnx')
        if os.path.exists(out_path):
            ok += 1
            continue
        try:
            input_name = get_input_name(mp)
            dr = RandomCalibrationDataReader(input_name, n_samples=a.calib_samples)

            # AMD-recommended CNN-for-NPU config
            quant_config = QuantizationConfig(
                calibrate_method=PowerOfTwoMethod.MinMSE,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                enable_npu_cnn=True,
                extra_options={'ActivationSymmetric': True},
            )
            config = Config(global_quant_config=quant_config)
            quantizer = ModelQuantizer(config)
            quantizer.quantize_model(mp, out_path, dr)
            ok += 1
            print(f'  [{i}/{len(models)}] {h[:10]} quantized')
        except Exception as e:
            err += 1
            print(f'  [{i}/{len(models)}] {h[:10]} ERROR {type(e).__name__}: {str(e)[:120]}')

    print(f'\nDone. {ok} int8 models in {a.out_dir} (errors: {err})')
    if err == 0:
        print('Next: time these on the NPU.')
    else:
        print('Some failed -- paste the error; may need Visual Studio for custom ops.')


if __name__ == '__main__':
    main()
