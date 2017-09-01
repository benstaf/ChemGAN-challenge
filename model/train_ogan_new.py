from __future__ import absolute_import, division, print_function
from builtins import range
from collections import OrderedDict
import os
import model
import numpy as np
import tensorflow as tf
import random
import time
import json
from gen_dataloader import Gen_Data_loader
from dis_dataloader import Dis_dataloader
#from text_classifier import TextCNN
from rollout_new import ROLLOUT
from target_lstm import TARGET_LSTM
# import io_utils
import pandas as pd
import mol_metrics
#import music_metrics
import sys
from tqdm import tqdm

#deepchem imports
import six
import deepchem
from graph_conv import GraphConvTensorGraph


if len(sys.argv) == 2:
    PARAM_FILE = sys.argv[1]
else:
    PARAM_FILE = 'exp_graph.json'
params = json.loads(open(PARAM_FILE).read(), object_pairs_hook=OrderedDict)
##########################################################################
#  Training  Hyper-parameters
##########################################################################

# Load metrics file
if params['METRICS_FILE'] == 'mol_metrics':
    mm = mol_metrics
elif params['METRICS_FILE'] == 'music_metrics':
    mm = music_metrics
else:
    raise ValueError('Metrics file unknown!')

PREFIX = params['EXP_NAME']
PRE_EPOCH_NUM = params['G_PRETRAIN_STEPS']
TRAIN_ITER = params['G_STEPS']  # generator
SEED = params['SEED']
BATCH_SIZE = params["BATCH_SIZE"]
TOTAL_BATCH = params['TOTAL_BATCH']
dis_batch_size = 64
dis_num_epochs = 3
dis_alter_epoch = params['D_PRETRAIN_STEPS']

##########################################################################
#  Generator  Hyper-parameters
##########################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
START_TOKEN = 0
SAMPLE_NUM = 6400
BIG_SAMPLE_NUM = SAMPLE_NUM * 5
D_WEIGHT = params['D_WEIGHT']

D = max(int(5 * D_WEIGHT), 1)
##########################################################################


##########################################################################
#  Discriminator  Hyper-parameters
##########################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

#============= Objective ==============

reward_func = mm.load_reward(params['OBJECTIVE'])


def make_reward(train_samples):

    def batch_reward(samples):
        decoded = [mm.decode(sample, ord_dict) for sample in samples]
        pct_unique = len(list(set(decoded))) / float(len(decoded))
        rewards = reward_func(decoded, train_samples)
        weights = np.array([pct_unique / float(decoded.count(sample))
                            for sample in decoded])

        return rewards * weights

    return batch_reward


def print_rewards(rewards):
    print('Rewards be like...')
    np.set_printoptions(precision=3, suppress=True)
    print(rewards)
    mean_r, std_r = np.mean(rewards), np.std(rewards)
    min_r, max_r = np.min(rewards), np.max(rewards)
    print('Mean: {:.3f} , Std:  {:.3f}'.format(mean_r, std_r),end='')
    print(', Min: {:.3f} , Max:  {:.3f}\n'.format(min_r, max_r))
    np.set_printoptions(precision=8, suppress=False)
    return
#=======================================
##########################################################################
train_samples = mm.load_train_data(params['TRAIN_FILE'])
char_dict, ord_dict = mm.build_vocab(train_samples)
NUM_EMB = len(char_dict)
DATA_LENGTH = max(map(len, train_samples))
MAX_LENGTH = params["MAX_LENGTH"]
to_use = [sample for sample in train_samples if mm.verified_and_below(
    sample, MAX_LENGTH)]
positive_samples = [mm.encode(sample, MAX_LENGTH, char_dict)
                    for sample in to_use]
POSITIVE_NUM = len(positive_samples)

df_pos=pd.DataFrame({'real/fake': [1]*len(to_use)   , 'smiles': to_use})  #positive samples for discriminator


print('Starting ObjectiveGAN for {:7s}'.format(PREFIX))
print('Data points in train_file {:7d}'.format(len(train_samples)))
print('Max data length is        {:7d}'.format(DATA_LENGTH))
print('Max length to use is      {:7d}'.format(MAX_LENGTH))
print('Avg length to use is      {:7f}'.format(
    np.mean([len(s) for s in to_use])))
print('Num valid data points is  {:7d}'.format(POSITIVE_NUM))
print('Size of alphabet is       {:7d}'.format(NUM_EMB))


mm.print_params(params)


##########################################################################

class Generator(model.LSTM):

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(0.002)  # ignore learning rate


def generate_samples(sess, trainable_model, batch_size, generated_num, verbose=False):
    #  Generated Samples
    generated_samples = []
    start = time.time()
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    end = time.time()
    if verbose:
        print('Sample generation time: %f' % (end - start))
    return generated_samples


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


# This is a hack. I don't even use LIkelihood data loader tbh
likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)


def pretrain(sess, generator, target_lstm, train_discriminator):
    # samples = generate_samples(sess, target_lstm, BATCH_SIZE, generated_num)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    gen_data_loader.create_batches(positive_samples)
    results = OrderedDict({'exp_name': PREFIX})

    #  pre-train generator
    print('Start pre-training...')
    start = time.time()
    for epoch in tqdm(range(PRE_EPOCH_NUM)):
        print(' gen pre-train')
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch == 10 or epoch % 40 == 0:
            samples = generate_samples(sess, generator, BATCH_SIZE, SAMPLE_NUM)
            likelihood_data_loader.create_batches(samples)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('\t test_loss {}, train_loss {}'.format(test_loss, loss))
            mm.compute_results(samples, train_samples, ord_dict, results)

    samples = generate_samples(sess, generator, BATCH_SIZE, SAMPLE_NUM)
    likelihood_data_loader.create_batches(samples)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)

    samples = generate_samples(sess, generator, BATCH_SIZE, SAMPLE_NUM)
    likelihood_data_loader.create_batches(samples)

    print('Start training discriminator...')
    for i in tqdm(range(dis_alter_epoch)):
        print(' discriminator pre-train')
        d_loss, acc = train_discriminator()
    end = time.time()
    print('Total time was {:.4f}s'.format(end - start))
    return


def save_results(sess, folder, name, results_rows=None):
    if results_rows is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv('{}_results.csv'.format(folder), index=False)
    # save models
    model_saver = tf.train.Saver()
    ckpt_dir = os.path.join(params['CHK_PATH'], folder)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(ckpt_dir, '{}.ckpt'.format(name))
    path = model_saver.save(sess, ckpt_file)
    print('Model saved at {}'.format(path))
    return


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # assert START_TOKEN == 0

    vocab_size = NUM_EMB
    dis_data_loader = Dis_dataloader()

    best_score = 1000
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM,
                          HIDDEN_DIM, MAX_LENGTH, START_TOKEN)
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE,
                              EMB_DIM, HIDDEN_DIM, MAX_LENGTH, 0)

    with tf.variable_scope('discriminator'):

#	batch_size=dis_batch_size
	cnn = GraphConvTensorGraph(n_tasks=1, batch_size=dis_batch_size, mode='classification')
	if not cnn.built:
	      cnn.build()

#indentation different 2 spaces
#  with cnn._get_tf("Graph").as_default():
#    manager = cnn._get_tf("Graph").as_default()
#    manager.__enter__()
    # Define Discriminator Training procedure
#    train_op = cnn._get_tf('train_op')

  # Define Discriminator Training procedure
    cnn_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]

    dis_global_step = tf.Variable(0, name="global_step", trainable=False)
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(
        cnn.loss.out_tensor, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(
        dis_grads_and_vars, global_step=dis_global_step)



    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    def train_discriminator():

	output_tensors = [cnn.outputs[0].out_tensor] #added from deepchem
    	fetches =  output_tensors + [dis_train_op, cnn.loss.out_tensor] 
        if D_WEIGHT == 0:
            return 0, 0

        negative_samples = generate_samples(
            sess, generator, BATCH_SIZE, POSITIVE_NUM)

        #  train discriminator
	#pos_smiles= [mm.decode(sample, ord_dict) for sample in positive_samples] #already defined in intro
	neg_smiles= [mm.decode(sample, ord_dict) for sample in negative_samples]
	#df_pos=pd.DataFrame({'real/fake': [1]*len(pos_smiles)   , 'smiles': pos_smiles}) #already defined in intro
	df_neg=pd.DataFrame({'real/fake': [0]*len(neg_smiles)   , 'smiles': neg_smiles}) 
	#df_total=df_pos.append(df_neg)
	df_total=pd.concat([df_pos,df_neg], ignore_index=True)

	
	def shuffle_by_batches(df, batch_size=dis_batch_size):
		l=len(df)
		l_batches=range(l)    
		l_batches=[l_batches[x*batch_size:(x+1)*batch_size] for x in range(np.ceil(l/batch_size))]
		permut=np.random.permutation(len(l_batches))
		l_new=[l_batches[n] for n in permut]
		l_new = [item for sublist in l_new for item in sublist] #flatten l_new
		df=df.reindex(np.array(l_new))
		df=df.reset_index(drop=True)   
		return df

	df_total=shuffle_by_batches(df_total,batch_size=dis_batch_size)    #shuffle rows by batch, see: Tip 4: https://github.com/soumith/ganhacks




	loader = deepchem.data.PandasLoader(tasks=['real/fake'], smiles_field="smiles", featurizer=deepchem.feat.ConvMolFeaturizer())
	train_dataset = loader.featurize(df_total, shard_size=8192)
	feed_dict_generator=cnn.default_generator(train_dataset, epochs=1)


	def create_feed_dict():
	      for d in feed_dict_generator:
	        feed_dict = {k.out_tensor: v for k, v in six.iteritems(d)}
	        feed_dict[cnn._training_placeholder] = 1.0
	        yield feed_dict

	def select_label(feed_dict):  #Select the output label layer in the feed dictionaty
		newkeys=[]
		for k in feed_dict.keys():
	             if 'Label' in k.name:
	                 newkeys.append(k)
		return newkeys[0]


	avg_loss, avg_acc, n_batches = 0.0, 0.0, 0

	for feed_dict in create_feed_dict():
	     fetched_values = sess.run(fetches, feed_dict=feed_dict)
	     loss = fetched_values[-1]
	     predicted_results=[np.argmax(x) for x in fetched_values[0]]
	     ground_truth= [np.argmax(x) for x in feed_dict[select_label(feed_dict)] ]
	     results= [ predicted_results[n] ==ground_truth[n] for n in range(len(predicted_results)) ]
	     accuracy= float(sum(results))/len(results)
	     n_batches += 1
	     ratio= (1/float(n_batches))
	     avg_loss= (1-ratio)*avg_loss + ratio*loss
	     avg_acc= (1-ratio)*avg_acc + ratio*accuracy
        print('\tD loss  :   {}'.format(avg_loss))
        print('\tAccuracy: {}'.format(avg_acc))
        return avg_loss, avg_acc

    # Pretrain is checkpointed and only execcutes if we don't find a checkpoint
    saver = tf.train.Saver()
    ckpt_dir = 'checkpoints_new/{}_pretrain'.format(PREFIX)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(ckpt_dir, 'pretrain_ckpt')
    if os.path.isfile(ckpt_file + '.meta') and params["LOAD_PRETRAIN"]:
        saver.restore(sess, ckpt_file)
        print('Pretrain loaded from previous checkpoint {}'.format(ckpt_file))
    else:
        if params["LOAD_PRETRAIN"]:
            print('\t* No pre-training data found as {:s}.'.format(ckpt_file))
        else:
            print('\t* LOAD_PRETRAIN was set to false.')
  	cnn._initialize_weights(sess, saver) #added from deepchem
        sess.run(tf.global_variables_initializer())
        pretrain(sess, generator, target_lstm, train_discriminator)
        path = saver.save(sess, ckpt_file)
        print('Pretrain finished and saved at {}'.format(path))

    # create reward function
    batch_reward = make_reward(train_samples)

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Reinforcement Training Generator...')
    results_rows = []
    for nbatch in tqdm(range(TOTAL_BATCH)):
        results = OrderedDict({'exp_name': PREFIX})
        if nbatch % 1 == 0 or nbatch == TOTAL_BATCH - 1:
            print('* Making samples')
            if nbatch % 10 == 0:
                gen_samples = generate_samples(
                    sess, generator, BATCH_SIZE, BIG_SAMPLE_NUM)
            else:
                gen_samples = generate_samples(
                    sess, generator, BATCH_SIZE, SAMPLE_NUM)
            likelihood_data_loader.create_batches(gen_samples)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('batch_num: {}'.format(nbatch))
            print('test_loss: {}'.format(test_loss))
            results['Batch'] = nbatch
            results['test_loss'] = test_loss

            if test_loss < best_score:
                best_score = test_loss
                print('best score: %f' % test_loss)

            # results
            mm.compute_results(gen_samples, train_samples, ord_dict, results)

        print('#########################################################################')
        print('-> Training generator with RL.')
        print('G Epoch {}'.format(nbatch))

        for it in range(TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(
                sess, samples, 16, cnn,  ord_dict, batch_reward, D_WEIGHT)
            nll = generator.generator_step(sess, samples, rewards)
            # results
            print_rewards(rewards)
            print('neg-loglike: {}'.format(nll))
            results['neg-loglike'] = nll
        rollout.update_params()

        # generate for discriminator
        print('-> Training Discriminator')
        for i in range(D):
            print('D_Epoch {}'.format(i))
            d_loss, accuracy = train_discriminator()
            results['D_loss_{}'.format(i)] = d_loss
            results['Accuracy_{}'.format(i)] = accuracy
        print('results')
        results_rows.append(results)
        if nbatch % params["EPOCH_SAVES"] == 0:
            save_results(sess, PREFIX, PREFIX + '_model', results_rows)

    # write results
    save_results(sess, PREFIX, PREFIX + '_model', results_rows)

    print('\n:*** FINISHED ***')
    return

if __name__ == '__main__':
    main()
