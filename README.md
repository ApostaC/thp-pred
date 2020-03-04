## LSTM for goodput prediction

### Usage:

python3 train-with-minmax.py \<load | train\> \<filename\>

`load`: load the trained model

`train`: train the model, use the specified configuration file

`filename`: When using `train` mode, this is the configuration file. When using `load` mode, this is the model file.

### folders:

`config/`: the folder for config files

`src/`: some useful scripts to monitor training progress

`models.`: to store models

### config parameters
`PREFIX`: logs will be dumped to `result-${PREFIX}`, and models will be written to `models/${PREFIX}-<epoch>.pt`.

`CKPT_EPOCH`: Trainer will dump models per `${CKPT_EPOCH}` epochs.

`BATCHSIZE`: the batchsize for training

`INPUTLEN`: the length of input traces used for training, in seconds.

`OUTPUTLEN`: the length of predictions, in seconds.

`LEARNING_RATE`: learning rate

`EPOCH`: stop at this epoch

### analysis scripts in `src/`
`analysis.sh`: It's an all-in-one script. Usage: `bash analysis.sh [time]`. It will draw some figures.
