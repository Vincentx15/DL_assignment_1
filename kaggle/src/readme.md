# Cats and Dogs Kaggle — Source

## Running an experiment

In order to run an experiment, just add a `config.json` file at the root of the project such as:

```json
{
  "name": "name of the experiment",
  "model": "model to use",
  "split": 0.9, 
  "size": 60,
  "random": "true",
  "seed": 1,
  "torch_seed": 1,
  "n_epochs": 100,
  "wall_time": 4,
  "batch_size": 10,
  "learning_rate": 1e-2,
  "log_interval": 10
}
```

Only `name` and `model` are required fields. The possible values for `model` are:

* `"baseline"` for [`baseline.Baseline`](models/baseline.py#L5)
* `"cnn"` for [`cnn.Network`](models/cnn.py#L5)
* `"resnet"` for [`resnet.ResNet`](models/resnet.py#L46)
* `"huge-cnn"` for [`cnn.HugeNetwork`](models/cnn.py#L64)
* `"meganet"` for [`meganet.MegaNet`](models/meganet.py#L4)

See the [docstrings for the `run_experiment()` method](run.py#L23) in `run.py` for more information on each field.

The best performing model as well as training logs are stored in the `results/` folder, under a subdirectory named after the `"name"` field of the configuration  file.
