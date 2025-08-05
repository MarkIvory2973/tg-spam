# TG Spam

Chinese Telegram spam messages classifier in PyTorch.

## Installation

Create an environment (Python **3.13**):

```bash
conda create -n tg-spam python=3.13
```

Install dependencies:

- For CPU:

  ```bash
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

- For NVIDIA GPU:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
  ```

```bash
pip install click matplotlib
```

Clone this repository:

```bash
git clone https://github.com/MarkIvory2973/tg-spam.git
```

## Usage

```bash
cd tg-spam
python src/cli.py train --batch-size 32 --learning-rate 0.001 --gamma 0.9 --epochs 25
python src/cli.py result
python src/cli.py prompt --epoch 5 --input "缺几个敢拼的兄弟😊，跟我干通宵，下个月路虎开回家，说到做到，看煮也"
python src/cli.py prompt --epoch 5 --input "网络是个神奇的地方hy2套cdn都出来了"
```

## Parameters

Train mode:

|Parameter|Required|Default|Description|
|:-|:-:|:-|:-|
|--root|-|./data/|Folder contains datasets and checkpoints|
|--batch-size|-|32|Batch size of dataset|
|--learning-rate|-|0.001|Learning rate of Adam|
|--gamma|-|0.9|Gamma of ExponentialLR|
|--epochs|-|25|Total epochs of training|

Result mode:

|Parameter|Required|Default|Description|
|:-|:-:|:-|:-|
|--root|-|./data/|Folder contains datasets and checkpoints|

Prompt mode:

|Parameter|Required|Default|Description|
|:-|:-:|:-|:-|
|--root|-|./data/|Folder contains datasets and checkpoints|
|--epoch|-|5|Epoch of model to use|
|--input|✓|-|Text to be classified|

## Result

||Accuracy (%)|
|:-:|-:|
|Eval|91.22%|

## References

[1] [Long Short-term Memory RNN](https://arxiv.org/abs/2105.06756)

[2] [reatiny/chinese-spam-10000](https://huggingface.co/datasets/reatiny/chinese-spam-10000)

[3] [paulkm/chinese_conversation_and_spam](https://huggingface.co/datasets/paulkm/chinese_conversation_and_spam)
