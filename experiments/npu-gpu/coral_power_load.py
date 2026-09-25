#!/usr/bin/env python3
"""
coral_power_load.py -- Put the Coral Edge TPU under STEADY, sustained load so a
USB power meter reads a stable draw. Runs one compiled _edgetpu.tflite model in a
tight continuous loop (no timing, no gaps) until you stop it with Ctrl+C.

Use this while watching your USB power meter's on-screen V/A/W + min/max/avg,
to characterize the Coral's INFERENCE power draw. Compare against idle (meter
reading with the Coral plugged in but this NOT running) to get active power.

Run on native Windows in the Coral env (with pycoral), same as your stage 3.

    python coral_power_load.py --model "path\\to\\some_edgetpu.tflite"
    # or point at your edgetpu folder and it picks one:
    python coral_power_load.py --edgetpu-dir npu_study\\...\\edgetpu

Tip: pick a LARGE model (cache-overflow) for peak power, and a small one for
lower draw, if you want to see the range. Default picks the largest file.
"""
import argparse, glob, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='a specific _edgetpu.tflite')
    ap.add_argument('--edgetpu-dir', default=None, help='folder to pick a model from')
    ap.add_argument('--pick', default='largest', choices=['largest', 'smallest', 'first'])
    ap.add_argument('--report-every', type=float, default=5.0,
                    help='seconds between throughput prints')
    a = ap.parse_args()

    try:
        from pycoral.utils.edgetpu import make_interpreter
        import numpy as np
    except ImportError:
        sys.exit('pycoral not found -- run in the Coral env on native Windows.')

    # resolve which model to run
    model = a.model
    if not model:
        if not a.edgetpu_dir:
            sys.exit('give --model or --edgetpu-dir')
        files = [f for f in glob.glob(os.path.join(a.edgetpu_dir, '*_edgetpu.tflite'))]
        if not files:
            sys.exit(f'no *_edgetpu.tflite in {a.edgetpu_dir}')
        if a.pick == 'largest':
            model = max(files, key=os.path.getsize)
        elif a.pick == 'smallest':
            model = min(files, key=os.path.getsize)
        else:
            model = sorted(files)[0]

    print(f'Model: {model}')
    print(f'Size: {os.path.getsize(model)/1024:.0f} KB')

    interp = make_interpreter(model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.randint(-128, 128, size=inp['shape'], dtype=inp['dtype'])
    interp.set_tensor(inp['index'], x)

    print('\n>>> Running CONTINUOUS inference. Watch your power meter now.')
    print('>>> Press Ctrl+C to stop.\n')

    n = 0
    t0 = time.time()
    t_last = t0
    try:
        while True:
            interp.invoke()
            n += 1
            now = time.time()
            if now - t_last >= a.report_every:
                rate = n / (now - t0)
                print(f'  {n} inferences | {rate:.0f}/sec sustained | '
                      f'{now-t0:.0f}s elapsed')
                t_last = now
    except KeyboardInterrupt:
        dt = time.time() - t0
        print(f'\nStopped. {n} inferences in {dt:.1f}s ({n/dt:.0f}/sec).')
        print('Read the min/avg/max power your meter recorded during this run.')


if __name__ == '__main__':
    main()
