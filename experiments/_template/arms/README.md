Arm files of this experiment, one `<arm>.yaml` each, in the layout of `configs/arms/`. Copy a core arm here and
change it to vary an arm; a file here shadows the core arm of the same name for this experiment only. An arm whose
backbone the core machine profiles do not list sets `encode_batch_size: {hub: <n>, cluster: <n>}`.
