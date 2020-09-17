# AMT-SIT

Dataset and code for our WWW 2020 paper "Anchored Model Transfer and Soft Instance Transfer for Cross-Task Cross-Domain Learning: A Study Through Aspect-Level Sentiment Classification"

## Requirement

- Python 3.6
- PyTorch 1.2.0
- NumPy 1.17.2
- GloVe pre-trained word vectors:
  - Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors).
  - Extract the [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) to the `/glove/` folder.

## Installation

An easy way to install this code with anaconda environment:

```bash
conda create -n repwalk python=3.6
conda activate repwalk
pip install -r requirements.txt
```

## Usage

Training the model:

```bash
python train.py --dataset [dataset]
```

Show help message and exit:

```bash
python train.py -h
```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{zheng2020anchored,
  title={Anchored Model Transfer and Soft Instance Transfer for Cross-Task Cross-Domain Learning: A Study Through Aspect-Level Sentiment Classification},
  author={Yaowei, Zheng and Richong, Zhang and Suyuchen, Wang and Samuel, Mensah and Yongyi, Mao},
  booktitle={Proceedings of The Web Conference 2020},
  pages={2754–2760},
  year={2020}
}
```

## License

MIT