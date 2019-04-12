# poetry-language-model
This project's goal was to build a poem generator for the 2019 National Poetry Day of Hungary. Click [here](http://oddnumberofeyes.com/versgenerator/) to generate your own poems! 

<p align="center">
  <img width="480" height="352" src="https://github.com/ben0it8/poetry-language-model/blob/master/pics/versgen.gif?raw=true">
</p>

---

To do this, I trained recurrent neural networks (RNNs) on poems modeling language statistically at subword level. The objective of training a language model is to predict the next word in a sequence given the words that precede it. Ultimately, we end up having a trained _generative model_, meaning that we can sample from it - in this case, new poems. Although these samples might resemble to human-written text they are produced by the model entirely. More on RNNs and LMs: [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

This repository contains:
* the [data](https://github.com/ben0it8/poetry-language-model/tree/master/data) used to train the models (one corpi for each poet),
* [Jupyter notebooks](https://github.com/ben0it8/poetry-language-model/tree/master/notebooks), implementing end-to-end language model training in PyTorch using the [SentencePiece tokenizer](https://github.com/google/sentencepiece),
* deployable Heroku [webapp](https://github.com/ben0it8/poetry-language-model/tree/master/heroku-app) in Flask.

For training I used [Google Colab](https://colab.research.google.com) to run the notebooks and the free GPU runtime. Using this repo you should be able to train your own language model on any text data.
