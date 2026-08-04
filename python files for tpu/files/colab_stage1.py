# ============================================================================
# NAS-Bench-101 -> Edge TPU latency study : STAGE 1 (Google Colab)
# ============================================================================
# Run this in Google Colab (https://colab.research.google.com, New Notebook).
# It samples architectures, builds Keras models, converts each to int8 TFLite,
# and zips them for download. Stages 2 (edgetpu_compiler) and 3 (Coral timing)
# run locally afterward on the downloaded files.
#
# Copy each block below into its own Colab cell (they are marked CELL 1, 2, ...).
# Runtime type can stay "CPU" -- int8 conversion is CPU-bound; no GPU needed.
# ============================================================================


# ============================================================================
# CELL 1 -- Config
# ============================================================================
SEED = 42               # fixed -> same sample every run (reproducible)
N_MODELS = 5000         # how many architectures to sample
NUM_BINS = 20           # stratify across param range into this many bins
REP_SAMPLES = 100       # representative-dataset size for int8 calibration
DRIVE_CHECKPOINT = True # mirror progress to Google Drive so a disconnect resumes

# Which dataset file. only108 is the smaller one (108-epoch results); it has all
# 423k architectures, which is all we need (we don't use the accuracy metrics).
TFRECORD_URL = 'https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord'
TFRECORD_FILE = 'nasbench_only108.tfrecord'

WORK_DIR = '/content/nasbench_study'
TFLITE_DIR = WORK_DIR + '/tflite'
CSV_PATH = WORK_DIR + '/results.csv'
DRIVE_DIR = '/content/drive/MyDrive/nasbench_study'   # used if DRIVE_CHECKPOINT

print('Config loaded. N_MODELS =', N_MODELS, '| SEED =', SEED)


# ============================================================================
# CELL 2 -- Download the dataset (~499 MB, a minute or two)
# ============================================================================
import os
os.makedirs(TFLITE_DIR, exist_ok=True)

if not os.path.exists(TFRECORD_FILE):
    print('Downloading NAS-Bench (~499 MB)...')
    os.system(f'wget -q --show-progress {TFRECORD_URL} -O {TFRECORD_FILE}')
    print('Done.')
else:
    print('Dataset already present.')
print('Size:', round(os.path.getsize(TFRECORD_FILE) / 1e6, 1), 'MB')


# ============================================================================
# CELL 3 -- (optional) Mount Google Drive for crash-safe checkpointing
# ============================================================================
# If the Colab session disconnects mid-run, re-running all cells resumes from
# the Drive copy instead of starting over. Skip this cell to disable.
if DRIVE_CHECKPOINT:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(DRIVE_DIR + '/tflite', exist_ok=True)
    # If a prior run left results on Drive, pull them back in to resume.
    if os.path.exists(DRIVE_DIR + '/results.csv'):
        os.system(f'cp {DRIVE_DIR}/results.csv {CSV_PATH}')
        os.system(f'cp -n {DRIVE_DIR}/tflite/*.tflite {TFLITE_DIR}/ 2>/dev/null')
        print('Resumed prior progress from Drive.')
    else:
        print('Drive mounted; no prior progress found (fresh start).')
else:
    print('Drive checkpointing disabled.')


# ============================================================================
# CELL 4 -- Robust NAS-Bench reader
# ============================================================================
# Two strategies, tried in order:
#   (A) the official nasbench API, lightly patched for TF2 (authentic param
#       counts straight from the dataset), and
#   (B) a dependency-free raw-JSON reader (each record is JSON: hash, epochs,
#       base64 adjacency, comma-joined ops, base64 metrics-we-ignore).
# Strategy B cannot hit the tf.train / protobuf walls because it imports nothing
# from nasbench and never touches the protobuf metrics blob.
import json, base64
import numpy as np

def _read_specs_raw(path):
    """Dependency-free: yield (hash, matrix, ops) from the raw JSON records."""
    import tensorflow as tf
    seen = set()
    it = tf.compat.v1.io.tf_record_iterator(path)  # TF2-safe iterator
    for serialized_row in it:
        module_hash, epochs, raw_adj, raw_ops, _raw_metrics = \
            json.loads(serialized_row.decode('utf-8'))
        if module_hash in seen:      # only108 has 3 repeats per arch; dedupe
            continue
        seen.add(module_hash)
        ops = raw_ops.split(',')
        dim = len(ops)
        adj = np.frombuffer(base64.b64decode(raw_adj), dtype=np.int8).reshape(dim, dim)
        yield module_hash, adj, ops

def load_all_specs(path):
    """Return list of (hash, matrix, ops). Tries official API, falls back to raw."""
    # ---- Strategy A: official API (patched) ----
    try:
        import sys, types
        import tensorflow as tf
        # tiny shims from nasbench issue #15 / #26 so the TF1 package imports
        if not hasattr(tf, 'train'):
            tf.train = types.ModuleType('train')
        for _n in ('SessionRunHook', 'SessionRunArgs', 'CheckpointSaverListener',
                   'NanLossDuringTrainingError'):
            if not hasattr(tf.train, _n):
                setattr(tf.train, _n, type(_n, (), {'__init__': lambda s,*a,**k: None}))
        if not hasattr(tf, 'python_io'):
            tf.python_io = tf.compat.v1.python_io
        from nasbench import api
        nb = api.NASBench(path)
        specs = []
        for h in nb.hash_iterator():
            fixed, _ = nb.get_metrics_from_hash(h)
            specs.append((h, np.array(fixed['module_adjacency'], dtype=np.int8),
                          list(fixed['module_operations']),
                          int(fixed['trainable_parameters'])))
        print(f'Loaded {len(specs)} specs via official nasbench API (with param counts).')
        return specs, True   # True = we have authentic param counts
    except Exception as e:
        print(f'Official API path failed ({type(e).__name__}: {e}).')
        print('Falling back to dependency-free raw-JSON reader...')

    # ---- Strategy B: raw JSON ----
    specs = [(h, adj, ops) for h, adj, ops in _read_specs_raw(path)]
    print(f'Loaded {len(specs)} specs via raw reader (param counts computed later).')
    return specs, False      # False = compute param counts from built models

specs, HAVE_PARAMS = load_all_specs(TFRECORD_FILE)
print('Total unique architectures:', len(specs), '| authentic params:', HAVE_PARAMS)


# ============================================================================
# CELL 5 -- The Keras builder (paste nasbench_keras.py contents here)
# ============================================================================
# IMPORTANT: paste the ENTIRE contents of nasbench_keras.py into this cell,
# replacing this comment. It defines build_keras_model() and
# convert_int8_tflite(), which the next cell calls. (It's the file from earlier
# in our conversation.)
#
# --- paste nasbench_keras.py below this line ---

# --- paste nasbench_keras.py above this line ---
print('Builder loaded:', 'build_keras_model' in dir(), 'convert_int8_tflite' in dir())


# ============================================================================
# CELL 6 -- Stratified sampling by parameter count
# ============================================================================
rng = np.random.default_rng(SEED)

if HAVE_PARAMS:
    # specs = (hash, matrix, ops, params)
    params = np.array([s[3] for s in specs])
else:
    # No param counts yet. Proxy param magnitude by edge count so the sample
    # still spans "small -> large" architectures; true params get recorded per
    # model during conversion below.
    params = np.array([int(s[1].sum()) for s in specs])
    print('Note: sampling stratified by edge-count proxy (raw reader path).')

bins = np.linspace(params.min(), params.max() + 1, NUM_BINS + 1)
which = np.digitize(params, bins)
per_bin = max(1, N_MODELS // NUM_BINS)

chosen = []
for b in range(1, NUM_BINS + 1):
    idx = np.where(which == b)[0]
    if len(idx):
        chosen.extend(rng.choice(idx, size=min(per_bin, len(idx)), replace=False).tolist())
if len(chosen) < N_MODELS:
    remaining = list(set(range(len(specs))) - set(chosen))
    chosen.extend(rng.choice(remaining,
                  size=min(N_MODELS - len(chosen), len(remaining)),
                  replace=False).tolist())
chosen = chosen[:N_MODELS]
print(f'Sampled {len(chosen)} architectures across {NUM_BINS} param bins (seed={SEED}).')


# ============================================================================
# CELL 7 -- Build + int8-convert each model, checkpointing as we go
# ============================================================================
import csv, time, tensorflow as tf

CSV_FIELDS = ['hash', 'params', 'tflite_file', 'tflite_bytes', 'status']

def load_done():
    if not os.path.exists(CSV_PATH):
        return {}
    with open(CSV_PATH, newline='') as f:
        return {r['hash']: r for r in csv.DictReader(f)}

def save_rows(rows):
    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader()
        for h in rows: w.writerow(rows[h])
    if DRIVE_CHECKPOINT:
        os.system(f'cp {CSV_PATH} {DRIVE_DIR}/results.csv')

rows = load_done()
already = sum(1 for r in rows.values() if r.get('status') == 'converted')
print(f'{already} already converted; resuming.\n')
t_start = time.time()

for count, i in enumerate(chosen, 1):
    spec = specs[i]
    h, matrix, ops = spec[0], spec[1], spec[2]
    if h in rows and rows[h].get('status') == 'converted' \
            and os.path.exists(os.path.join(TFLITE_DIR, f'{h}.tflite')):
        continue

    row = {'hash': h, 'params': (spec[3] if HAVE_PARAMS else ''),
           'tflite_file': '', 'tflite_bytes': '', 'status': ''}
    try:
        model = build_keras_model(matrix, ops)
        if not HAVE_PARAMS:
            row['params'] = model.count_params()   # record true params from build
        tfl = convert_int8_tflite(model, rep_samples=REP_SAMPLES)
        fname = f'{h}.tflite'
        with open(os.path.join(TFLITE_DIR, fname), 'wb') as f:
            f.write(tfl)
        row['tflite_file'] = fname
        row['tflite_bytes'] = len(tfl)
        row['status'] = 'converted'
    except Exception as e:
        row['status'] = f'error:{type(e).__name__}'   # skip-and-log
        print(f'  [{count}/{len(chosen)}] {h[:10]} SKIP {row["status"]}')

    rows[h] = row
    tf.keras.backend.clear_session()   # prevent memory creep over thousands of builds

    if count % 25 == 0:
        save_rows(rows)
        if DRIVE_CHECKPOINT:
            os.system(f'cp -n {TFLITE_DIR}/*.tflite {DRIVE_DIR}/tflite/ 2>/dev/null')
        rate = count / (time.time() - t_start)
        eta = (len(chosen) - count) / rate / 60
        ok = sum(1 for r in rows.values() if r.get('status') == 'converted')
        print(f'  [{count}/{len(chosen)}] {ok} ok | {rate*60:.1f}/min | ETA {eta:.0f} min')

save_rows(rows)
if DRIVE_CHECKPOINT:
    os.system(f'cp -n {TFLITE_DIR}/*.tflite {DRIVE_DIR}/tflite/ 2>/dev/null')
ok = sum(1 for r in rows.values() if r.get('status') == 'converted')
print(f'\nStage 1 complete: {ok}/{len(chosen)} models converted to int8 TFLite.')


# ============================================================================
# CELL 8 -- Zip everything and download
# ============================================================================
print('Zipping...')
os.system(f'cd {WORK_DIR} && zip -q -r nasbench_stage1.zip tflite results.csv')
zip_path = WORK_DIR + '/nasbench_stage1.zip'
print('Zip size:', round(os.path.getsize(zip_path) / 1e6, 1), 'MB')

from google.colab import files
files.download(zip_path)   # browser download prompt
print('Download started. Unzip locally, then run stages 2 & 3 on the .tflite files.')
