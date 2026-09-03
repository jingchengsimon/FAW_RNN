# Clutter protocol

The publication-oriented Clutter training is complete. Its retained figures and minimal numeric
inputs live in `results/save/` and `results/save_data/`; reproduction scripts remain in
`utils/analysis/`. Historical grids are not part of the active project tree.

The active data-scale behavior campaign uses
`experiments/clutter/amarel/submit_clutter_data_scale_formal.sh`. Each scale has 60 task IDs:
`task_id = model_index * 10 + seed - 1`, with model order
`rnn,lstm,gru,gawf,mamba,s5` and seeds 1-10. Runs use the fixed best6 hyperparameters, 150 epochs,
`patience=0`, a shared `40h-uint8` validation set, and the current standard uint8 pipeline.
Results are isolated below
`results/data/clutter/runs/data_scale/clutter_formal_4scale_ep150/<scale>/<model>-seedNN/`.

Generate the required `4h-uint8`, `10h-uint8`, and `20h-uint8` training splits with
`experiments/clutter/amarel/submit_clutter_data_scale_generate_uint8.sh`. The generator uses the
same exclusive-switch stimulus settings as 40h, records seed 42, and publishes each NPY/TSV pair
plus its completion manifest directly beside the retained `40h-uint8` files. Scale training keeps
using the existing `40h-uint8` validation split, so shorter validation/test copies are not made.
