# Born networks

Training code for BornNet, LinearBornNet, and BornAE from
[Neural network enhancement of the Born approximation](https://arxiv.org/abs/2610.32764),
by J. Ma, A. Desai, T. Lähivaara, and P. Monk.

Each network takes the real and imaginary parts of a 100×100 scattering-data
array and reconstructs the real contrast μ = n² − 1 on a 100×100 image grid.

| Model | Data-to-image map | Training loss |
| --- | --- | --- |
| BornNet | Fixed Born inverse | Final and intermediate image MSE |
| LinearBornNet | Learned rank-1,024 linear map | Final and intermediate image MSE |
| BornAE | LinearBornNet encoder and Born-based field decoder | Image MSE + physical-field MSE + 10⁻⁴ mean absolute contrast |

`models.py` defines the networks, `blocks.py` their shared layers, and
`train.py` the training loop. Data, fixed operators, trained weights, forward
solvers, and evaluation/plotting scripts are not included.

## Run

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --python 3.11
uv run train.py --model LinearBornNet --data data --operators data/operators.npz --output runs/linear
uv run train.py --model BornNet --data data --operators data/operators.npz --output runs/born
uv run train.py --model BornAE --data data --operators data/operators.npz --output runs/ae
```

CUDA is used when available. Add `--amp` for bfloat16 neural computations,
or `--device cpu` for CPU execution. The default step count, batch size,
and learning rate follow the near-field experiments in Section 5:
25,000 Adam updates, batches of 16, and a learning rate of 3×10⁻⁴.
Gradients are clipped at norm 1. Every 1,000 steps, the script evaluates
the mean relative contrast error on the validation set and saves the
best model to `best.pt`.

## Data and operators

Supply separate `data/train.npz` and `data/val.npz` files. Each contains:

- `fields`: complex array `(N, 10000)` of scattered-field measurements.
- `truth`: real contrast μ = n² − 1, with shape `(N, 1, 100, 100)`.
  Validation targets must have nonzero contrast norm.

Measurements and images are flattened in C order, with all real measurements
stacked before all imaginary measurements. The forward matrix has measurement
rows and image columns, and the inverse has image rows and measurement
columns. Use this same ordering in the data and operators.
Supply noiseless training and validation data for the paper experiments.

`operators.npz` contains the fixed operators in unnormalized field units:

- `inverse`: real `(10000, 20000)` matrix, required only for BornNet.
- `forward`: real `(20000, 10000)` matrix, required only for BornAE.
- `data_scale`: optional positive scalar used to normalize the inputs.

For a complex Born matrix B, the real forward matrix stacks Re(B) above
Im(B). For an existing complex inverse T whose reconstruction is Re(Ty),
the equivalent real inverse is `[Re(T), −Im(T)]`. Alternatively, supply
the real-contrast Tikhonov inverse directly. The trainer does not build or
regularize these matrices.

Without a supplied scale, inputs are unnormalized. For near-field data,
use the RMS of the real and imaginary **training** channels, either in
`operators.npz` or through `--data-scale`. Pass the same scale to all models
(LinearBornNet can read it from `--operators` without loading a matrix).
Contrast targets are never scaled. The inverse layer applies sK, the BornAE
decoder applies B/s, and the field loss multiplies normalized-field MSE by
s² to retain physical-field units. Far-field inputs can use s = 1.

Fixed operators are excluded from `best.pt`. To reload BornNet or BornAE,
construct the model with the same operators and saved `data_scale`, then
load the saved `model` state dictionary and call `model.eval()`. Divide
input channels by the saved scale before inference. `model.encode(x)[0]`
returns the contrast without running the BornAE decoder.
