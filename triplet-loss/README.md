## TODOs
For triplet loss
- [x] check the embeddings
- visualize the embeddings
- use a large dataset
- l2 norm or cosine distance, decide which one is better for similarity representation
- figure out whether switching between batch_all and batch_hard is necessary

For next steps
- replace the build_model with AlexNet
- [x] turn the net into a classification task

## Requirements

We recommend using python3 and a virtual environment.
The default `venv` should be used, or `virtualenv` with `python3`.

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements_cpu.txt
```

If you are using a GPU, you will need to install `tensorflow-gpu` so do:
```bash
pip install -r requirements_gpu.txt
```

## Triplet loss

|![triplet-loss-img] |
|:--:|
| *Triplet loss on two positive faces (Obama) and one negative face (Macron)* |



The interesting part, defining triplet loss with triplet mining can be found in [`model/triplet_loss.py`](models/triplet_loss.py).

Everything is explained in the [blog post][blog].

To use the "batch all" version, you can do:
```python
from model.triplet_loss import batch_all_triplet_loss

loss, fraction_positive = batch_all_triplet_loss(labels, embeddings, margin, squared=False)
```

In this case `fraction_positive` is a useful thing to plot in TensorBoard to track the average number of hard and semi-hard triplets.

To use the "batch hard" version, you can do:
```python
from model.triplet_loss import batch_hard_triplet_loss

loss = batch_hard_triplet_loss(labels, embeddings, margin, squared=False)
```

## Training on MNIST

To run a new experiment called `base_model`, do:
```bash
python train.py --model_dir experiments/base_model
```

You will first need to create a configuration file like this one: [`params.json`](experiments/base_model/params.json).
This json file specifies all the hyperparameters for the model.
All the weights and summaries will be saved in the `model_dir`.

Once trained, you can visualize the embeddings by running:
```bash
python visualize_embeddings.py --model_dir experiments/base_model
```

And run tensorboard in the experiment directory:
```bash
tensorboard --logdir experiments/base_model
```

Here is the result ([link][embeddings-gif] to gif):

|![embeddings-img] |
|:--:|
| *Embeddings of the MNIST test images visualized with T-SNE (perplexity 25)* |



## Test

To run all the tests, run this from the project directory:
```bash
pytest
```

To run a specific test:
```bash
pytest model/tests/test_triplet_loss.py
```


## Resources
- [CS230 code template][CS230]
- [triplet loss tensorflow implementation][triplet implementation]
- [Blog post][blog] explaining this project.
- Source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].
- [Facenet paper][facenet] introducing online triplet mining
- Detailed explanation of online triplet mining in [*In Defense of the Triplet Loss for Person Re-Identification*][in-defense]
- Blog post by Brandom Amos on online triplet mining: [*OpenFace 0.2.0: Higher accuracy and halved execution time*][openface-blog].
- Source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].
- The [coursera lecture][coursera] on triplet loss

[CS230]: https://cs230.stanford.edu
[triplet implementation]: https://github.com/omoindrot/tensorflow-triplet-loss
[blog]: https://omoindrot.github.io/triplet-loss
[triplet-types-img]: https://omoindrot.github.io/assets/triplet_loss/triplets.png
[triplet-loss-img]: https://omoindrot.github.io/assets/triplet_loss/triplet_loss.png
[online-triplet-loss-img]: https://omoindrot.github.io/assets/triplet_loss/online_triplet_loss.png
[embeddings-img]: https://omoindrot.github.io/assets/triplet_loss/embeddings.png
[embeddings-gif]: https://omoindrot.github.io/assets/triplet_loss/embeddings.gif
[openface-blog]: http://bamos.github.io/2016/01/19/openface-0.2.0/
[facenet]: https://arxiv.org/abs/1503.03832
[in-defense]: https://arxiv.org/abs/1703.07737
[tf-triplet-loss]: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss
[coursera]: https://www.coursera.org/learn/convolutional-neural-networks/lecture/HuUtN/triplet-loss
