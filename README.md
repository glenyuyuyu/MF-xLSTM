# MF-xLSTM
Multi-frequency xLSTM implementation for NeuralHydrology.

The code is provided as a lightweight patch for [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology). It replaces NeuralHydrology's original `x_lstm.py` implementation so that `x_lstm` can be used with multi-frequency inputs, following the same general training and output workflow as NeuralHydrology's built-in MTS-LSTM models.

## Files

```text
.
├── README.md
├── x_lstm.py
└── config_mfxlstm_example.yml
```

- `x_lstm.py`: MF-xLSTM model implementation.
- `config_mfxlstm_example.yml`: Example NeuralHydrology configuration file.

## Usage

First, install NeuralHydrology and the required `xlstm` package.

Then locate the installed NeuralHydrology package:

```bash
python -c "import neuralhydrology, pathlib; print(pathlib.Path(neuralhydrology.__file__).parent)"
```

This gives the local NeuralHydrology path, referred to below as `<NH_PATH>`.

Back up the original xLSTM implementation:

```bash
cp <NH_PATH>/modelzoo/x_lstm.py <NH_PATH>/modelzoo/x_lstm_original.py
```

Replace it with the MF-xLSTM implementation from this repository:

```bash
cp x_lstm.py <NH_PATH>/modelzoo/x_lstm.py
```

Then open:

```text
<NH_PATH>/modelzoo/__init__.py
```

and remove or comment out `x_lstm` from the `SINGLE_FREQ_MODELS` list.

After this modification, the model can be used in a NeuralHydrology configuration file with:

```yaml
model: x_lstm
```

An example configuration is provided in:

```text
config_mfxlstm_example.yml
```

## Notes

This implementation is not part of the official NeuralHydrology release. It is provided as a patch-based implementation for reproducing the MF-xLSTM experiments in the accompanying manuscript.

The model uses NeuralHydrology's standard multi-frequency input and output workflow. In a daily-hourly setup, it can be trained and evaluated in the same general way as NeuralHydrology's built-in multi-frequency LSTM (MTS-LSTM) model.
