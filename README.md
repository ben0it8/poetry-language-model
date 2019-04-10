# poetry-language-model
This project's goal was to build a poem generator for the 2019 National Poetry Day in Hungary. To do this, I trained recurrent neural networks (RNNs) modelling language at subword level. Language models (LMs) aim at predicting the next word in a sequence given the words that precede it, ultimately yielding a _generative_ model, meaning that we can sample from it. Although these samples can be very similar to human-written text they are made up by the language model. More on RNNs and LMs: [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

This repository contains:
* the [data](https://github.com/ben0it8/poetry-language-model/tree/master/data) used to train the models (one corpi for each poet),
* end-to-end training scripts (Jupyter Notebooks),
* web application in Flask, which was deployed on [Heroku](https://poetry-day.herokuapp.com/).

For training I used [Google Colab](https://colab.research.google.com) and the free GPU runtime.

To generate your own poems, click [here](http://oddnumberofeyes.com/versgenerator/)!
