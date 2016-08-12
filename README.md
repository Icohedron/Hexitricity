# Asynchronous Advantage Actor-Critic (A3C) with Monte Carlo Tree Search and Hex

To train a new network, remove the following files:
* saved_networks/checkpoint
* saved_networks/Hex9x9-v0-Hexitricty.checkpoint
* saved_networks/Hex9x9-v0-Hexitricty.checkpoint.meta
* saved_networks/tf_summaries/events.out.tfevents.*

Then execute async.py using the command
```
python async.py
```
To evaluate the network, set async.py's TRAIN variable to False and run the file again.

## Dependencies:
* [tqdm] (https://pypi.python.org/pypi/tqdm)
* [gym] (https://gym.openai.com/)
* [tensorflow] (https://www.tensorflow.org/)
* [numpy] (http://www.numpy.org/)
