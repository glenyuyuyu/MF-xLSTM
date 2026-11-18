# MF-xLSTM and MF-LSTM
Multi-frequency xLSTM and LSTM implementations for NeuralHydrology.

The code is provided as lightweight patches for [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology). MF-xLSTM replaces NeuralHydrology's original `x_lstm.py` implementation so that `x_lstm` can be used with multi-frequency inputs, following the same general training and output workflow as NeuralHydrology's built-in MTS-LSTM models. MF-LSTM replaces the original `mtslstm.py` implementation and follows the same multi-frequency input-packing and output workflow as MF-xLSTM, with a standard LSTM backbone instead of xLSTM.

## Files

```text
.
├── README.md
├── x_lstm.py
├── mtslstm.py
└── config_mfxlstm_example.yml
```

- `x_lstm.py`: MF-xLSTM model implementation.
- `mtslstm.py`: MF-LSTM model implementation.
- `config_mfxlstm_example.yml`: Example NeuralHydrology configuration file.

## Usage

First, install NeuralHydrology and its required dependencies. For MF-xLSTM, also install the required `xlstm` package.

Then locate the installed NeuralHydrology package:

```bash
python -c "import neuralhydrology, pathlib; print(pathlib.Path(neuralhydrology.__file__).parent)"
```

This gives the local NeuralHydrology path, referred to below as `<NH_PATH>`.

Run the replacement commands below from the directory containing the provided model scripts.

### MF-xLSTM

Before replacing the original xLSTM implementation, it is recommended to keep a backup by renaming it, for example to `original_x_lstm.py`:

```bash
mv <NH_PATH>/modelzoo/x_lstm.py <NH_PATH>/modelzoo/original_x_lstm.py
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

### MF-LSTM

Before replacing the original MTS-LSTM implementation, it is recommended to keep a backup by renaming it, for example to `old_mtslstm.py`:

```bash
mv <NH_PATH>/modelzoo/mtslstm.py <NH_PATH>/modelzoo/old_mtslstm.py
```

Replace it with the MF-LSTM implementation from this repository:

```bash
cp mtslstm.py <NH_PATH>/modelzoo/mtslstm.py
```

No changes to `<NH_PATH>/modelzoo/__init__.py` or its `SINGLE_FREQ_MODELS` list are required for MF-LSTM. NeuralHydrology already loads the `MTSLSTM` class for `mtslstm`, and `mtslstm` is not included in `SINGLE_FREQ_MODELS`.

After replacement, the model can be used in a NeuralHydrology configuration file with:

```yaml
model: mtslstm
```

Use the same multi-frequency configuration structure as MF-xLSTM, including `use_frequencies` and frequency-specific `seq_length` and `predict_last_n` settings. The `xlstm_*` backbone settings are not used by MF-LSTM.

## Notes

These implementations are not part of the official NeuralHydrology release. They are provided as patches for reproducing the MF-xLSTM and MF-LSTM experiments in the accompanying manuscript.

> Yu, Q., Tolson, B., & Lin, J. (2027). Exploring xLSTM for large-sample hourly rainfall–runoff modelling. *Environmental Modelling & Software, 207*, 107193. https://doi.org/10.1016/j.envsoft.2026.107193

Both models use NeuralHydrology's standard multi-frequency input and output workflow. In a daily-hourly setup, they can be trained and evaluated in the same general way as NeuralHydrology's built-in multi-timescale LSTM (MTS-LSTM) model.

