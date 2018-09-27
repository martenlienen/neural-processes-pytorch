# Neural Processes in pytorch

An implementation of neural processes as described in [this
paper](https://arxiv.org/abs/1807.01622) by Garnelo et al. Two other
implementations I found are [Kaspar Märtens' version in
R](https://github.com/kasparmartens/NeuralProcesses) and [Chris Ormandy's blog
post](https://chrisorm.github.io/NGP.html) both of which were helpful to figure
out some implementation details. [deepmind's notebook on conditional neural
processes](https://github.com/deepmind/conditional-neural-process/blob/master/conditional_neural_process.ipynb),
their first publication, is of course also interesting.

The model is able to perform n-dimensional regression because I wanted to
recreate the MNIST image completion experiment which you can find in the
[jupyter notebook](./mnist_completion.ipynb).
