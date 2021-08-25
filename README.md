# Conv-Seq2Seq
This is my attempt to implement the "[Convolutional Sequence to Sequence
Learning](https://arxiv.org/pdf/1705.03122.pdf)" paper published by FacebookAI
in 2017 using PyTorch. My implementation has been done fully using PyTorch and
PyTorchAudio.

The aim of this paper is to use a Convolution networks as a new architecture
for translation. One of the major defects of Seq2Seq models is that it can’t
process words in parallel. For a large corpus of text, this increases the time
spent translating the text. CNNs can help us solve this problem as we can 
parallelize them. Accordinng to the paper, Conv Seq2Seq outperformed the
Attention model on both WMT’14 English-German and WMT’14 English-French
translation while achieving faster results.


<div align="Center">
    <a href="https://arxiv.org/pdf/1705.03122.pdf"> <img src="docs/cover.png" width=500> </a>
</div>


# Prerequisites

To install the dependencies for this project, you need to run the following command:

```
pip install -r requirements.txt
```

**NOTE**

If you are using `spacy` as a tokenizer, don't forget to download the source and target language models. For example, the small language model for English can be download by running the following command:
```
python -m spacy download en_core_web_sm
```

