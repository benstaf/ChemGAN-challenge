import numpy as np
import six
import tensorflow as tf

from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot, from_one_hot
#from deepchem.models.tensorgraph.graph_layers import WeaveLayer, WeaveGather, \
#    Combine_AP, Separate_AP, DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, DAGGather, DTNNExtract
from deepchem.models.tensorgraph.layers import Dense, Concat, SoftMax, SoftMaxCrossEntropy, GraphConv, BatchNorm, \
    GraphPool, GraphGather, WeightedError #, BatchNormalization
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator


class GraphConvTensorGraph(TensorGraph):

  def __init__(self, n_tasks, **kwargs):
    """
        Parameters
        ----------
        n_tasks: int
          Number of tasks

    """
    self.n_tasks = n_tasks
    kwargs['use_queue'] = False
    super(GraphConvTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    """
    Building graph structures:
    """
    self.atom_features = Feature(shape=(None, 75))
    self.degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
    self.membership = Feature(shape=(None,), dtype=tf.int32)

    self.deg_adjs = []
    for i in range(0, 10 + 1):
      deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
      self.deg_adjs.append(deg_adj)
    gc1 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[self.atom_features, self.degree_slice, self.membership] +
        self.deg_adjs)
    batch_norm1 = BatchNorm(in_layers=[gc1])
    gp1 = GraphPool(in_layers=[batch_norm1, self.degree_slice, self.membership]
                    + self.deg_adjs)
    gc2 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[gp1, self.degree_slice, self.membership] + self.deg_adjs)
    batch_norm2 = BatchNorm(in_layers=[gc2])
    gp2 = GraphPool(in_layers=[batch_norm2, self.degree_slice, self.membership]
                    + self.deg_adjs)
    dense = Dense(out_channels=128, activation_fn=None, in_layers=[gp2])
    batch_norm3 = BatchNorm(in_layers=[dense])
    gg1 = GraphGather(
        batch_size=self.batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[batch_norm3, self.degree_slice, self.membership] +
        self.deg_adjs)

    costs = []
    self.my_labels = []
    for task in range(self.n_tasks):
      if self.mode == 'classification':
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[gg1])

        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.my_labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)

#	self.y_pred= tf.argmax(softmax, 1)  #calculer y_pred


      if self.mode == 'regression':
        regression = Dense(out_channels=1, activation_fn=None, in_layers=[gg1])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.my_labels.append(label)
        cost = L2Loss(in_layers=[label, regression])
        costs.append(cost)

    entropy = Concat(in_layers=costs, axis=-1)
    self.my_task_weights = Weights(shape=(None, self.n_tasks))

#    print('entropy is '+ str(type(entropy)))
 #   print('weights are '+ str(type(self.my_task_weights)))

    loss = WeightedError(in_layers=[entropy, self.my_task_weights])
    self.set_loss(loss)

  #  print('loss has type '+ str(type(loss))) 

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(
              self.batch_size, pad_batches=True, deterministic=True)):
        d = {}
        for index, label in enumerate(self.my_labels):
          if self.mode == 'classification':
            d[label] = to_one_hot(y_b[:, index])
          if self.mode == 'regression':
            d[label] = np.expand_dims(y_b[:, index], -1)
        d[self.my_task_weights] = w_b
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        d[self.atom_features] = multiConvMol.get_atom_features()
        d[self.degree_slice] = multiConvMol.deg_slice
        d[self.membership] = multiConvMol.membership
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          d[self.deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
        yield d

  def predict_proba_on_generator(self, generator, transformers=[]):
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, self.last_checkpoint)
        out_tensors = [x.out_tensor for x in self.outputs]
        results = []
        for feed_dict in generator:
          feed_dict = {
              self.layers[k.name].out_tensor: v
              for k, v in six.iteritems(feed_dict)
          }
          result = np.array(sess.run(out_tensors, feed_dict=feed_dict))
          if len(result.shape) == 3:
            result = np.transpose(result, axes=[1, 0, 2])
          if len(transformers) > 0:
            result = undo_transforms(result, transformers)
          results.append(result)
        return np.concatenate(results, axis=0)

  def evaluate(self, dataset, metrics, transformers=[], per_task_metrics=False):
    if not self.built:
      self.build()
    return self.evaluate_generator(
        self.default_generator(dataset),
        metrics,
        labels=self.my_labels,
        weights=[self.my_task_weights])

  def predict_on_smiles(self, smiles, transformers):
    max_index = len(smiles)
    num_batches = max_index // self.batch_size

    y_ = []
    for i in range(num_batches):
      smiles_batch = smiles[i * self.batch_size:(i + 1) * self.batch_size]
      y_.append(self.predict_on_smiles_batch(smiles_batch, transformers))
    smiles_batch = smiles[num_batches * self.batch_size:max_index]
    y_.append(self.predict_on_smiles_batch(smiles_batch, transformers))

    return np.concatenate(y_, axis=1)

  def predict_on_smiles_batch(self, smiles, transformers=[]):
    featurizer = ConvMolFeaturizer()
    convmols = featurize_smiles_np(smiles, featurizer)

    n_smiles = convmols.shape[0]
    n_tasks = len(self.outputs)

    dataset = NumpyDataset(X=convmols, y=None, n_tasks=n_tasks)
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    y_ = self.predict_on_generator(generator, transformers)

    return y_.reshape(-1, n_tasks)[:n_smiles]




 # Accuracy
#        with tf.name_scope("accuracy"):
 #           correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
  #          self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


