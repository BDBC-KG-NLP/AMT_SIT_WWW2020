# AMT-SIT

Code and dataset for our WWW 2020 paper "Anchored Model Transfer and Soft Instance Transfer for Cross-Task Cross-Domain Learning: A Study Through Aspect-Level Sentiment Classification"

You can download the paper via: [[Github]](paper.pdf) [[DOI]](https://doi.org/10.1145/3366423.3380034).

## Abstract

Supervised learning relies heavily on readily available labelled data to infer an effective classification function. However, proposed methods under the supervised learning paradigm are faced with the scarcity of labelled data within domains, and are not generalized enough to adapt to other tasks. Transfer learning has proved to be a worthy choice to address these issues, by allowing knowledge to be shared across domains and tasks. In this paper, we propose two transfer learning methods Anchored Model Transfer (AMT) and Soft Instance Transfer (SIT), which are both based on multi-task learning, and account for model transfer and instance transfer, and can be combined into a common framework. We demonstrate the effectiveness of AMT and SIT for aspect-level sentiment classification showing the competitive performance against baseline models on benchmark datasets. Interestingly, we show that the integration of both methods AMT+SIT achieves state-of-the-art performance on the same task.

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
conda create -n amtsit python=3.6
conda activate amtsit
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

## Acknowledgments

This work is supported partly by the National Natural Science Foundation of China, by the Beijing Advanced Innovation Center for Big Data and Brain Computing (BDBC), by State Key Laboratory of Software Development Environment, by the Beijing S&T Committee and by the Fundamental Research Funds for the Central Universities.

## Contact

hiyouga [AT] buaa [DOT] edu [DOT] cn

## License

MIT
