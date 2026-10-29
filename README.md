# Born networks

This repository has the PyTorch code that trains BornNet, LinearBornNet, and
BornAE, the three networks in
[Neural network enhancement of the Born approximation](https://arxiv.org/abs/2610.32764)
by J. Ma, A. Desai, T. Lähivaara, and P. Monk.

Each network reads the real and imaginary parts of a 100×100 array of
scattering measurements. It returns an image of the contrast μ = n² − 1 on a
100×100 grid, where n is the refractive index. A U-Net corrects the
measurements, an inverse layer maps them to an intermediate image, and a
residual network refines that image. All three networks use the same U-Net and
residual network. They differ in the inverse layer and the training loss.

| Network | Inverse layer | Training loss |
| --- | --- | --- |
| BornNet | Fixed regularized Born inverse | Mean squared error of the final and intermediate images |
| LinearBornNet | Learned linear map of rank at most 1,024 | Mean squared error of the final and intermediate images |
| BornAE | Learned linear map of rank at most 1,024 | Mean squared error of the final image and of the decoded measurements, plus 10⁻⁴ times the mean absolute contrast |

BornAE also has a decoder that runs only during training. The decoder applies
the forward Born matrix to the final image and corrects the result with a small
convolutional network. The training loss compares the decoder output with the
input measurements.

`models.py` defines the networks, and `blocks.py` defines their layers.
`operators.py` builds the operator file, and `train.py` trains the networks.

## Training

Install [uv](https://docs.astral.sh/uv/). For near-field data in `data`, these
commands build the operator file and train the three networks:

```sh
uv sync --python 3.11
uv run operators.py nearfield --data data --output data/operators.npz
uv run train.py --model BornNet --data data --operators data/operators.npz --output runs/born
uv run train.py --model LinearBornNet --data data --operators data/operators.npz --output runs/linear
uv run train.py --model BornAE --data data --operators data/operators.npz --output runs/ae
```

For far-field data, replace the second command with
`uv run operators.py farfield --output data/operators.npz`.

`train.py` uses Adam and clips the gradient norm at 1. Its defaults match the
near-field runs in Section 5 of the paper: 25,000 steps, batch size 16,
learning rate 3×10⁻⁴, and a validation check every 1,000 steps. At each check,
it computes the mean relative contrast error on the validation set. It saves
the weights to `best.pt` in the output directory whenever this error is the
lowest so far.

`train.py` runs on CUDA when a GPU is available. `--device cpu` forces the CPU. On
a GPU with bfloat16 support, `--amp` runs the learned layers under bfloat16
autocast.

## Data

The `--data` directory holds `train.npz` and `val.npz`. Each file has two
arrays.

- `fields` is a complex array of shape `(N, 10000)`. Each row holds the
  measurements of one object as a 100×100 array flattened in C order. The
  table under Operators gives the axes of that array.
- `truth` is a real array of shape `(N, 1, 100, 100)` that holds the contrasts.

Every validation contrast must be nonzero, because `train.py` divides each
error by the norm of the true contrast. In the paper, the training and
validation data are noiseless. Only the test data has added noise.

## Operators

BornNet and BornAE need two fixed matrices. `train.py` loads them from the
operator file, which you pass with `--operators`. The file holds three arrays.

- `inverse` is the real `(10000, 20000)` matrix K that BornNet uses.
- `forward` is the real `(20000, 10000)` matrix A that BornAE uses.
- `data_scale` is the input scale s.

Both matrices list image pixels in C order and put the real parts of the
measurements before the imaginary parts. For a complex Born matrix B, A stacks
Re(B) above Im(B).

`operators.py` builds the operator file for each of the two setups in the
paper. Both setups use wavenumber 16 and time dependence exp(−iωt). The first
axis of each image is y and the second is x. Your data must use the same
geometry.

| | `farfield` | `nearfield` |
| --- | --- | --- |
| Measurements | Incident and observation directions at the same 100 equally spaced angles from −π to π, including both ends | 100 antennas equally spaced on a circle of radius 1.8, starting at angle 0. Each antenna is both a source and a receiver. |
| Rows of the measurement array | Observation directions | Sources |
| Columns of the measurement array | Incident directions | Receivers |
| Image points | 100 equally spaced values from −1 to 1, including both ends, in x and in y | Centers of the cells of a 100×100 grid on [−1, 1]² |
| Inverse K | Truncated pseudo-inverse of B | Tikhonov inverse of A |
| Scale s | 1 | Root mean square of the real and imaginary parts of the training measurements |

The paper's far-field BornNet uses the pseudo-inverse B⁺ of B. The
pseudo-inverse discards the singular values of B below 1.19×10⁻³ times the
largest, which leaves 559 of them. The real form of B⁺ is
K = [Re(B⁺), −Im(B⁺)], so Ky = Re(B⁺y).

For the near field, `operators.py` tries ten Tikhonov parameters and keeps the
one with the lowest mean relative contrast error on `val.npz`. It also reads
`train.npz` to compute s. The far-field build needs about 13 GB of memory,
and the near-field build needs about 10 GB.

## Input scaling

`train.py` divides the measurements by s. The `--data-scale` option sets s.
Without it, `train.py` reads `data_scale` from the operator file. If neither
gives a value, s is 1.

Store K and A without the factor s, because the networks apply it. The BornNet
inverse layer applies sK. The BornAE decoder applies A/s. The BornAE loss
multiplies the decoder's mean squared error by s², which gives that error in
the original units. `train.py` does not scale the contrasts.

For LinearBornNet, `train.py` reads only `data_scale` from the operator file.
Pass the same file to all three networks so that they use the same s.

## Loading a checkpoint

The `BornNet` class builds all three networks, and its `kind` argument selects
one. `best.pt` holds the weights under `model`, along with `kind` and
`data_scale`. It does not hold K or A. To reload a network, construct
`BornNet(kind, inverse=..., forward=..., data_scale=...)` with the matrices
from training, load the weights, and call `model.eval()`.

The input is an `(N, 2, 100, 100)` tensor of the real and imaginary parts of
the measurements. Divide it by `data_scale` first, as in training.
`model.encode(x)` returns the final and intermediate images and skips the
BornAE decoder.
