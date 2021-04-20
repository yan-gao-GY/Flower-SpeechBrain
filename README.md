# Flower-SpeechBrain

This repository integrates [Flower](https://flower.dev) and [SpeechBrain](https://speechbrain.github.io) to achieve training ASR models in FL setting.

## Requiments
* Flower and SpeechBrain can be installed based on the instruction on their GitHub repo ([Flower GitHub](https://github.com/adap/flower) and [SpeechBrain GitHub](https://github.com/speechbrain/speechbrain)).
* PyTorch version >= 1.5.0
* Python version >= 3.7

## Examples
The server should be launched first.
``` bash
python server.py --data_path /path/to/dataset --tr_path /path/to/train.tsv --test_path /path/to/test.tsv --tr_add_path /path/to/train_add.tsv --weight_strategy num
```

After obtaining server's IP address, the clients can be launched. Here is an example to start one client.
``` bash
python client.py --cid 0 --server_address xx.xx.xx.xx --data_path /path/to/dataset --tr_path /path/to/train.tsv --dev_path /path/to/dev.tsv --pre_train_model_path /path/to/model.ckpt --tokenizer_path /path/to/tokenizer
```
Note that the `tokenizer_path` can be the `train.csv` file from model pre-training.