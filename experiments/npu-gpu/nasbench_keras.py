"""
nasbench_keras.py

Turn a NAS-Bench-101 (matrix, ops) spec into a fully int8-quantizable Keras
model that reproduces the NAS-Bench skeleton, so the trainable-parameter count
matches the number reported in the dataset.

The channel-projection logic (compute_vertex_channels, prune, truncate) is
ported faithfully from the original google-research/nasbench implementation
(via the NNI reproduction) so the built cell is structurally identical to what
the dataset evaluated -- same interior channel counts, same add/concat/project
behaviour, hence the same parameter count.

Only dependency: tensorflow (2.x).

Op labels (must match the strings used in the NAS-Bench API):
    INPUT       = 'input'
    OUTPUT      = 'output'
    CONV1X1     = 'conv1x1-bn-relu'
    CONV3X3     = 'conv3x3-bn-relu'
    MAXPOOL3X3  = 'maxpool3x3'
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

# NAS-Bench-101 fixed skeleton hyperparameters (from the paper / config).
# These are what make the param counts line up with the dataset.
DEFAULT_STEM_CHANNELS = 128
DEFAULT_NUM_STACKS = 3
DEFAULT_MODULES_PER_STACK = 3
NUM_LABELS = 10  # CIFAR-10


# --------------------------------------------------------------------------
# Graph logic ported from the original NAS-Bench implementation.
# --------------------------------------------------------------------------
def prune(matrix, ops):
    """Remove vertices not connected to both input and output, then compact.

    Mirrors ModelSpec pruning in the original repo. Returns (matrix, ops).
    """
    matrix = np.array(matrix, dtype=int)
    ops = list(ops)
    num_vertices = matrix.shape[0]

    # Reachability within num_vertices steps.
    connections = np.linalg.matrix_power(matrix + np.eye(num_vertices, dtype=int),
                                         num_vertices)
    visited_from_input = set(i for i in range(num_vertices) if connections[0, i])
    visited_from_output = set(i for i in range(num_vertices) if connections[i, -1])
    extraneous = set(range(num_vertices)) - (visited_from_input & visited_from_output)

    if len(extraneous) > num_vertices - 2:
        raise ValueError('Graph is not connected input->output; invalid spec.')

    matrix = np.delete(matrix, list(extraneous), axis=0)
    matrix = np.delete(matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
        del ops[index]
    return matrix, ops


def compute_vertex_channels(input_channels, output_channels, matrix):
    """Channel count at every vertex. Faithful port of the original.

    Interior vertices take the max channels of the vertices they feed into.
    Output channels are divided among vertices feeding the output; uneven
    division gives some vertices +1 channel.
    """
    num_vertices = matrix.shape[0]
    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        return vertex_channels

    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]

    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity checks from the original.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2

    return vertex_channels


# --------------------------------------------------------------------------
# Keras op primitives.
# --------------------------------------------------------------------------
def _truncate(tensor, channels):
    """Drop extra channels (channels guaranteed non-decreasing => diff is 0 or 1)."""
    input_channels = tensor.shape[-1]
    if input_channels < channels:
        raise ValueError('input channels < target channels in truncate')
    if input_channels == channels:
        return tensor
    assert input_channels - channels == 1
    return layers.Lambda(lambda x: x[..., :channels])(tensor)


def _conv_bn_relu(x, channels, kernel_size, name):
    x = layers.Conv2D(channels, kernel_size, padding='same', use_bias=False,
                      name=f'{name}_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    return x


def _vertex_op(x, op_label, channels, name):
    if op_label == CONV1X1:
        return _conv_bn_relu(x, channels, 1, name)
    if op_label == CONV3X3:
        return _conv_bn_relu(x, channels, 3, name)
    if op_label == MAXPOOL3X3:
        # stride 1, same padding -> preserves spatial dims, channels unchanged
        return layers.MaxPool2D(pool_size=3, strides=1, padding='same',
                                name=f'{name}_maxpool')(x)
    raise ValueError(f'Unknown op label: {op_label}')


def _projection(x, channels, name):
    """1x1 conv-bn-relu projection used on the input tensor of a vertex/output."""
    return _conv_bn_relu(x, channels, 1, name)


# --------------------------------------------------------------------------
# Build a single module (cell) from a pruned spec.
# --------------------------------------------------------------------------
def build_module(x, matrix, ops, output_channels, name):
    """Construct one NAS-Bench module. x has `output_channels`-ish input channels.

    Follows the original: tensors into a vertex are summed; into the output are
    concatenated; the input tensor is always projected before being added.
    """
    input_channels = x.shape[-1]
    num_vertices = matrix.shape[0]

    if num_vertices == 2:
        # input -> output directly; project to match.
        return _projection(x, output_channels, f'{name}_proj_io')

    vertex_channels = compute_vertex_channels(input_channels, output_channels, matrix)

    tensors = [x]  # index 0 is the module input
    for t in range(1, num_vertices - 1):
        add_in = []
        for src in range(1, t):
            if matrix[src, t]:
                add_in.append(_truncate(tensors[src], vertex_channels[t]))
        if matrix[0, t]:
            add_in.append(_projection(tensors[0], vertex_channels[t],
                                      f'{name}_v{t}_inproj'))

        if len(add_in) == 1:
            vertex_input = add_in[0]
        else:
            vertex_input = layers.Add(name=f'{name}_v{t}_add')(add_in)

        vertex_out = _vertex_op(vertex_input, ops[t], vertex_channels[t],
                                f'{name}_v{t}')
        tensors.append(vertex_out)

    # Output vertex: concat all fan-in, add projected input if connected.
    out_srcs = [src for src in range(1, num_vertices - 1)
                if matrix[src, num_vertices - 1]]

    if np.sum(matrix[:, num_vertices - 1]) == 1 and len(out_srcs) == 1:
        # single fan-in, no concat needed
        output = tensors[out_srcs[0]]
    elif len(out_srcs) == 1 and matrix[0, num_vertices - 1]:
        output = tensors[out_srcs[0]]
    else:
        concat_tensors = [tensors[src] for src in out_srcs]
        if len(concat_tensors) == 1:
            output = concat_tensors[0]
        else:
            output = layers.Concatenate(axis=-1,
                                        name=f'{name}_out_concat')(concat_tensors)

    if matrix[0, num_vertices - 1]:
        proj = _projection(tensors[0], output_channels, f'{name}_out_inproj')
        output = layers.Add(name=f'{name}_out_add')([output, proj])

    return output


# --------------------------------------------------------------------------
# Assemble the full network with the NAS-Bench skeleton.
# --------------------------------------------------------------------------
def build_keras_model(matrix, ops,
                      input_shape=(32, 32, 3),
                      stem_channels=DEFAULT_STEM_CHANNELS,
                      num_stacks=DEFAULT_NUM_STACKS,
                      modules_per_stack=DEFAULT_MODULES_PER_STACK,
                      num_labels=NUM_LABELS):
    """Build the full CIFAR-10 network for a NAS-Bench spec.

    Skeleton: stem 3x3 conv -> [stack of modules -> downsample]xN -> GAP -> dense.
    Channels double after each downsample (128 -> 256 -> 512), matching the
    dataset's fixed skeleton so parameter counts align.
    """
    matrix, ops = prune(matrix, ops)

    inp = tf.keras.Input(shape=input_shape, name='input')
    # Stem
    x = _conv_bn_relu(inp, stem_channels, 3, 'stem')

    channels = stem_channels
    for stack in range(num_stacks):
        # Downsample between stacks (not before the first).
        if stack > 0:
            x = layers.MaxPool2D(pool_size=2, strides=2, padding='same',
                                 name=f'downsample{stack}')(x)
            channels *= 2
        for m in range(modules_per_stack):
            x = build_module(x, matrix, ops, channels,
                             name=f's{stack}_m{m}')

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    out = layers.Dense(num_labels, name='logits')(x)

    return tf.keras.Model(inp, out, name='nasbench_model')


# --------------------------------------------------------------------------
# int8 TFLite conversion for the Edge TPU.
# --------------------------------------------------------------------------
def convert_int8_tflite(model, rep_samples=200, input_shape=(32, 32, 3)):
    """Full-integer int8 quantization. Returns tflite bytes.

    Uses random representative data (fine for latency work -- calibration only
    needs correctly-shaped inputs, not real labels).
    """
    def rep_gen():
        for _ in range(rep_samples):
            data = np.random.rand(1, *input_shape).astype(np.float32)
            yield [data]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


if __name__ == '__main__':
    # Inception-like example from the NAS-Bench README.
    matrix = [[0, 1, 1, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0]]
    ops = [INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]

    model = build_keras_model(matrix, ops)
    model.summary()
    print('\nTotal params:', model.count_params())

    print('\nConverting to int8 TFLite...')
    tflite_bytes = convert_int8_tflite(model, rep_samples=20)
    print('TFLite size (bytes):', len(tflite_bytes))
    with open('nasbench_model_int8.tflite', 'wb') as f:
        f.write(tflite_bytes)
    print('Wrote nasbench_model_int8.tflite')
