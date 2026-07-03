# SigScope metric workspace

SigScope is a PyQt desktop application for computing statistics from complex
signal matrices and exporting the selected outputs to CSV.

## Run

```powershell
pip install -r requirements.txt
python run.py
```

## Supported input

Input files may be ordinary two-dimensional NumPy arrays or poured
three-channel arrays. A poured pixel is decoded with:

```text
value = channel_0 + channel_1 + channel_2
```

For example, `[255, 255, 190]` becomes `700`. The decoded two-dimensional
matrix is used for statistics while the original three-channel matrix remains
available in the preview.

- **Amplitude only:** enables amplitude moments and pulse-pair power.
- **Phase only:** enables phase, coherence, Doppler, chirp, and spectral
  metrics. Internally, unit amplitude is used where a complex representation
  is required.
- **Amplitude + phase:** load the two matching arrays separately. Their shapes
  must be identical.
- **Complex:** loads one native complex NumPy matrix or a stacked
  amplitude+phase array. Supported stack layouts include `[H,W,2]`, `[2,H,W]`,
  `[H,W,6]`, `[6,H,W]`, and `[2,H,W,3]`. Six-channel layouts are interpreted
  as three poured amplitude channels followed by three poured phase channels.

The internal convention is `z[range, time]`. Clear **Axis 0 contains range
bins** when the source matrix is stored as `[time, range]`.

## Compute and export

In the **Compute & Export** tab:

1. load one or more supported channels;
2. tick the exact metric outputs required;
3. choose row analysis (along time) or column analysis (along range);
4. click **Compute selected metrics & save CSV…**.

The CSV has an index column followed by one column for each selected metric.
Metric names are prefixed by their statistic group so similarly named outputs
remain unambiguous.

## Visualizations

The **Stat Visualizations** tab uses the latest computed result. Click **Add
empty visualization** to create another plot and choose its metric from the
dropdown. Plots can be added or removed independently.

There is no row-selection mode. All computations use the full loaded matrix.
