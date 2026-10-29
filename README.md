# Born networks

PyTorch training code for BornNet, LinearBornNet, and BornAE, the three
networks introduced in
[Neural network enhancement of the Born approximation](https://arxiv.org/abs/2610.32764)
by J. Ma, A. Desai, T. Lähivaara, and P. Monk.

Each network takes the real and imaginary parts of a 100×100 array of
scattering measurements and reconstructs the contrast μ = n² − 1 on a 100×100
grid, where n is the refractive index. A U-Net corrects the measurements, an
inverse layer maps them to an intermediate image, and a residual network
refines that image. The U-Net and refiner are the same in all three networks,
which differ in the inverse layer and the training loss:

| Model | Inverse layer | Training loss |
| --- | --- | --- |
| BornNet | Fixed regularized Born inverse | Final-image MSE + intermediate-image MSE |
| LinearBornNet | Learned linear map of rank ≤ 1,024 | Final-image MSE + intermediate-image MSE |
| BornAE | Same as LinearBornNet | Final-image MSE + field MSE + 10⁻⁴ × mean absolute contrast |

The BornAE field loss uses a decoder that runs only during training. It applies
the forward Born matrix to the reconstruction and corrects the resulting field
with a small convolutional network.

`models.py` defines the networks, `blocks.py` their shared layers, and
`train.py` the training loop.

## Training

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --python 3.11
uv run train.py --model BornNet --data data --operators data/operators.npz --output runs/born
uv run train.py --model LinearBornNet --data data --operators data/operators.npz --output runs/linear
uv run train.py --model BornAE --data data --operators data/operators.npz --output runs/ae
```

The script trains with Adam and clips gradients to norm 1. Its defaults match
the near-field experiments in Section 5 of the paper: 25,000 steps, batch size
16, learning rate 3×10⁻⁴, and a validation check every 1,000 steps. Each check
computes the mean relative contrast error on the validation set, and `best.pt`
in the output directory keeps the model with the lowest error.

CUDA is used when available, and `--device cpu` forces the CPU. On a GPU with
bfloat16 support, `--amp` runs the learned layers under bfloat16 autocast.

## Data

The `--data` directory contains `train.npz` and `val.npz`, each with two arrays:

- `fields`: complex `(N, 10000)` scattered-field measurements. Each row is a
  100×100 measurement array flattened in C order.
- `truth`: real `(N, 1, 100, 100)` contrasts μ.

Validation contrasts must be nonzero, since the validation error is relative to
their norm. The paper trains and validates on noiseless data and adds noise
only to test data.

## Operators

BornNet and BornAE need fixed operators, passed through `--operators` as an
`.npz` file:

- `inverse`: real `(10000, 20000)` matrix K, used by BornNet.
- `forward`: real `(20000, 10000)` matrix A, used by BornAE.
- `data_scale`: optional input scale s.

Both matrices index image pixels in C order and place the real parts of the
measurements before the imaginary parts. For a complex Born matrix B, A stacks
Re(B) above Im(B). The far-field experiments reconstruct Re(Ty) with a complex
Tikhonov inverse T, which corresponds to K = [Re(T), −Im(T)]. The near-field
experiments use the Tikhonov inverse of A as K.

## Input scaling

The script divides the measurements by a scale s. It reads s from
`--data-scale`, falling back to `data_scale` in the operators file and then
to 1. The operators stay in the original field units, because the networks
account for s: the BornNet inverse layer applies sK, the BornAE decoder applies
A/s, and the BornAE field loss is s² times the normalized field MSE, which
equals the field MSE in the original units. Contrasts are not scaled.

The far-field experiments use s = 1. The near-field experiments use one s for
all three models: the RMS of the real and imaginary parts of the training
measurements. LinearBornNet reads only `data_scale` from `--operators`.

## Loading a checkpoint

All three models are built by the `BornNet` class, selected by its `kind`
argument. `best.pt` stores the weights under `model`, together with `kind` and
`data_scale`, but not the fixed operators. To reload a model, construct
`BornNet(kind, inverse=..., forward=..., data_scale=...)` with the operators
used in training, load the saved weights, and call `model.eval()`. The input is
an `(N, 2, 100, 100)` tensor of real and imaginary parts, divided by
`data_scale` as in training. `model.encode(x)` returns the reconstruction and
the intermediate image without running the BornAE decoder.
