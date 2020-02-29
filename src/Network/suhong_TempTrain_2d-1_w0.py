import os,sys
import numpy as np
import random

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

NCPU = 4

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

##################### input parameters ###############################
FILEDIR		= sys.argv[1]
INDIR		= sys.argv[2]  # native npz dir
nb_epochs	= int(sys.argv[3])
n2d_layers	= int(sys.argv[4])
n2d_filters	= int(sys.argv[5])
method		= sys.argv[6]


# FILEDIR      = "/dl/suhong/project/TemptrRosetta/DB/DB13989"
# INDIR        = "/dl/yangjy/project/distance/npz"       # native npz dir
# nb_epochs    = 1
# n2d_layers   = 11
# n2d_filters  = 8

test_file    = "%s/test_lst"%(FILEDIR)
all_file     = "%s/list"%(FILEDIR)
TEMPDIR      = "%s/temp_npz1"%(FILEDIR)  # template npz dir

# test set
with open(test_file) as f:
	test_ids = f.read().splitlines()

# all set: containing train and test list
with open(all_file) as f:
	IDs = f.read().splitlines()

maxseq = 20000   
minseq = 1

dmax = 20.0
dmin = 2.0
nbins = 36
kmin = 6

bins = np.linspace(dmin, dmax, nbins+1)
bins180 = np.linspace(0.0, np.pi, 13)
bins360 = np.linspace(-np.pi, np.pi, 25)

def npz_loader(ID, DIR = INDIR, DIR2 = TEMPDIR):
	name = DIR + '/' + ID + '.npz'
	name_temp = DIR2 + '/' + ID + '_T01' + '.npz'
	npz = np.load(name) 
	npz_temp = np.load(name_temp)  # temp npz
	# print(name_temp)  #test

	#### native npz objective ####
	# dist
	dist = npz['dist'] 
	dist[dist<0.001] = 999.9
	# bin distance matrix
	dbin = np.digitize(dist, bins).astype(np.uint8)
	dbin[dbin > nbins] = 0

	# bin omega
	obin = np.digitize(npz['omega'], bins360).astype(np.uint8)
	obin[dbin == 0] = 0

	# bin theta
	tbin = np.digitize(npz['theta_asym'], bins360).astype(np.uint8)
	tbin[dbin == 0] = 0

	# bin phi
	pbin = np.digitize(npz['phi_asym'], bins180).astype(np.uint8)
	pbin[dbin == 0] = 0

	#### template npz objective ####
	# dist (Cb-Cb)
	d = npz_temp['dist'] 
	d[d<0.001] = 999.9
	tdbin = np.digitize(d, bins).astype(np.uint8)
	tdbin[tdbin > nbins] = 0

	# omega
	om = npz_temp['omega']
	tobin = np.digitize(om, bins360).astype(np.uint8)
	tobin[tdbin == 0] = 0

	# theta_asym
	th_asym = npz_temp['theta_asym']
	ttbin = np.digitize(th_asym, bins360).astype(np.uint8)
	ttbin[tdbin == 0] = 0

	# phi_asym
	ph_asym = npz_temp['phi_asym']
	tpbin = np.digitize(ph_asym, bins180).astype(np.uint8)
	tpbin[tdbin == 0] = 0

	# template weight
	weight = npz_temp['weight']

	return {'mask1d' : npz['mask1d'],
			'msa' : npz['msa'][:maxseq,:],
			'dbin' : dbin,
			'obin' : obin,
			'tbin' : tbin,
			'pbin' : pbin,
			'dist' : dist,
			'bbpairs' : npz['bbpairs'],
			'd' : tdbin,
			'om' : tobin,
			'th_asym' : ttbin,
			'ph_asym' : tpbin,
			'weight' : weight,
			'label' : ID}

# judge whether or not len(native.npz) == len(temp.npz) 
remove_ids = []
for i in range(len(IDs)):
	nat_npz = np.load(INDIR + '/' + IDs[i] + '.npz')
	tem_npz = np.load(TEMPDIR + '/' + IDs[i] + '_T01' + '.npz')
	nat_len = nat_npz['dist'].shape[0]
	tem_len = tem_npz['dist'].shape[0]
	if nat_len != tem_len:
		remove_ids.append(IDs[i])

for i in range(len(remove_ids)):
	print(remove_ids[i])

sub_test_ids = list(set(test_ids)-set(remove_ids))
sub_IDs = list(set(IDs)-set(remove_ids))
train_ids = list(set(sub_IDs) - set(sub_test_ids))

train = []
test = [] 
s = 0.0

# load data in parallel using NCPU threads
with ProcessPoolExecutor(max_workers=NCPU) as executor:
	futures_test = [executor.submit(npz_loader, (ID)) for ID in sub_test_ids]
	results_test = concurrent.futures.wait(futures_test)
	for f in futures_test:
		test.append(f.result())
		for key,arr in test[-1].items():
			if key != "label":
				s += arr.nbytes

with ProcessPoolExecutor(max_workers=NCPU) as executor:
	futures_train = [executor.submit(npz_loader, (ID)) for ID in train_ids]
	results_train = concurrent.futures.wait(futures_train)
	for f in futures_train:
		train.append(f.result())
		for key,arr in train[-1].items():
			if key != "label":
				s += arr.nbytes
				
print('# {:d} proteins, {:.2f}gb'.format(len(train)+len(test), s / 1024**3))
print('# Training set: ', len(train))
print('# Testing set: ', len(test))

# save model
# PREFIX = "Train" + str(len(train)) + "_Test" + str(len(test))
PREFIX = method
CHK = "/dl/suhong/project/TemptrRosetta/models/model." + PREFIX + '-layer%s'%(n2d_layers) + '-filter%s'%(n2d_filters) + '-epoch%s'%(nb_epochs)

sys.stdout.flush()

import tensorflow as tf

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.50)
)

lr           = 0.0001
l2_coef      = 0.0001

window2d     = 3
maxlen       = 260  
wmin         = 0.8
ns           = 21

# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
	beff = tf.reduce_sum(w)
	f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
	h_i = tf.reduce_sum( -f_i * tf.math.log(f_i), axis=1)
	return tf.concat([f_i, h_i[:,None]], axis=1)

# randomly subsample the alignment
def subsample_msa(msa):

	nr, nc = msa.shape  

	# do not subsample shallow MSAs
	if nr < 10:
		return msa

	# number of sequences to subsample
	n = int(10.0**np.random.uniform(np.log10(nr))-10.0)
	if n <= 0:
		return np.reshape(msa[0], [1,nc])

	# n unique indices
	sample = sorted(random.sample(range(1, nr-1), n))

	# stack first sequence with n randomly selected ones
	msa_new = np.vstack([msa[0][None,:], msa[1:][sample]])

	return msa_new.astype(np.uint8)

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):

	with tf.name_scope('reweight'):
		id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff
		id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
		id_mask = id_mtx > id_min
		w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
	return w

def fast_dca(msa1hot, weights, penalty = 4.5):

	nr = tf.shape(msa1hot)[0]
	nc = tf.shape(msa1hot)[1]
	ns = tf.shape(msa1hot)[2]

	with tf.name_scope('covariance'):
		x = tf.reshape(msa1hot, (nr, nc * ns))
		num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
		mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
		x = (x - mean) * tf.sqrt(weights[:,None])
		cov = tf.matmul(tf.transpose(x), x)/num_points


	with tf.name_scope('inv_convariance'):
		cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
		inv_cov = tf.linalg.inv(cov_reg)
		
		# reduce entropy effect
		#wd = tf.linalg.diag_part(w)
		#w = w/tf.sqrt(wd[None,:]*wd[:,None])
		
		x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
		x2 = tf.transpose(x1, [0,2,1,3])
		features = tf.reshape(x2, (nc, nc, ns * ns))
		
		x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
		apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
		contacts = (x3 - apc) * (1-tf.eye(nc))

	return tf.concat([features, contacts[:,:,None]], axis=2)

def cont_acc(pred, dist, frac = 1.0):
	nc = dist.shape[0]
	if nc < 24:
		return 0.0
	w = np.sum(pred[0,:,:,1:13], axis=-1)
	idx = np.array([[i,j,w[i,j]] for i in range(nc) for j in range(i+24,nc)])
	n = int(nc*frac)
	top = idx[np.argsort(idx[:,2])][-n:, :2].astype(int)
	ngood = np.sum([dist[r[0],r[1]]<8.0 for r in top])
	return 1.0 * ngood / n

def get_fdict(item, flag):
	# native npz objective
	L = item['msa'].shape[1]
	msa_ = subsample_msa(item['msa'])
	theta_ = item['tbin']
	omega_ = item['obin']
	phi_ = item['pbin']
	dist_ = item['dbin']
	mask1d_ = item['mask1d']
	bbpairs_ = item['bbpairs']

	# template npz objective
	d_ = item['d']
	om_ = item['om']
	th_ = item['th_asym']
	ph_ = item['ph_asym']
	w_ = item['weight']

	start = 0
	stop = L
	nres = L

	# slice vertically if sequence is too long
	if L > maxlen:
		nres = np.random.randint(150,maxlen)
		start = np.random.randint(L-maxlen+1)
		stop = start + nres

	item['start'] = start
	item['stop'] = stop

	feed_dict = {
		ncol: nres,
		nrow: msa_.shape[0],
		msa: msa_[:,start:stop],
		theta: theta_[start:stop,start:stop],
		omega: omega_[start:stop,start:stop],
		phi: phi_[start:stop,start:stop],
		dist: dist_[start:stop,start:stop],
		mask1d: mask1d_[start:stop],
		bbpairs: bbpairs_[start:stop,start:stop],
		d: d_[start:stop,start:stop],
		om: om_[start:stop,start:stop],
		th: th_[start:stop,start:stop],
		ph: ph_[start:stop,start:stop],
		w_temp: w_,
		is_train: flag}
	
	return feed_dict

activation = tf.nn.elu
conv2d = tf.layers.conv2d


with tf.Graph().as_default():
	
	with tf.name_scope('input'):
		ncol = tf.compat.v1.placeholder(dtype=tf.int32, shape=())
		nrow = tf.compat.v1.placeholder(dtype=tf.int32, shape=())
		msa = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
		mask1d = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None))
		bbpairs = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
		dist = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
		omega = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
		theta = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
		phi = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
		d = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None))
		om = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None))
		th = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None))
		ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None))
		w_temp = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
		is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

	######################## convert inputs to 1-hot ######################
	msa1hot  = tf.one_hot(msa, ns, dtype=tf.float32)
	dist1hot = tf.one_hot(dist, nbins+1, dtype=tf.float32)
	omega1hot = tf.one_hot(omega, 25, dtype=tf.float32)
	theta1hot = tf.one_hot(theta, 25, dtype=tf.float32)
	phi1hot = tf.one_hot(phi, 13, dtype=tf.float32)
	bb1hot   = tf.one_hot(bbpairs, 3, dtype=tf.float32)

	########################### collect features ###########################
	###### msa ########
	# 1d
	w = reweight(msa1hot, wmin)
	neff = tf.reduce_sum(w)
	f1d_seq = msa1hot[0,:,:20]
	f1d_pssm = msa2pssm(msa1hot, w)
	f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
	f1d = tf.expand_dims(f1d, axis=0)
	f1d = tf.reshape(f1d, [1,ncol,42])
	# 2d
	f2d_dca = tf.cond(nrow>1, lambda: fast_dca(msa1hot, w), lambda: tf.zeros([ncol,ncol,442], tf.float32))
	f2d_dca = tf.expand_dims(f2d_dca, axis=0)

	f2d_query = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
					tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
					f2d_dca], axis=-1) 

	###### template ########
	# 2d
	tem_d = tf.expand_dims(tf.cast(d[:,:,None],dtype=tf.float32),axis=0)
	tem_om = tf.expand_dims(tf.cast(om[:,:,None],dtype=tf.float32),axis=0)
	tem_th = tf.expand_dims(tf.cast(th[:,:,None],dtype=tf.float32),axis=0)
	tem_ph = tf.expand_dims(tf.cast(ph[:,:,None],dtype=tf.float32),axis=0)

	f2d_temp = tf.concat([tem_d,tem_om,tem_th,tem_ph],axis=-1)  # (1,L,L,4) tf.uint8

	############# connect query and one template###################
	#f2d = tf.concat([f2d_query * 1.0, f2d_temp * w_temp],axis=-1)
	#f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42+4])

	############## connect 1d and 2d (trRosetta)###############
	f2d = f2d_query
	f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])
	
	############# only template #######################
	# f2d = tf.reshape(f2d_temp, [1,ncol,ncol,4])

	layers2d = [f2d]
	layers2d.append(conv2d(layers2d[-1], n2d_filters, 1, padding='SAME'))
	layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
	layers2d.append(activation(layers2d[-1]))

	# D.Jones resnet
	dilation = 1
	for _ in range(n2d_layers):
		layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
		layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
		layers2d.append(activation(layers2d[-1]))
		layers2d.append(tf.keras.layers.Dropout(rate=0.15)(layers2d[-1], training=is_train))
		layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
		layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
		layers2d.append(activation(layers2d[-1] + layers2d[-7]))
		dilation *= 2
		if dilation > 16:
			dilation = 1

	# loss on theta
	logits_theta = conv2d(layers2d[-1], 25, 1, padding='SAME')
	out_theta = tf.nn.softmax_cross_entropy_with_logits_v2(labels=theta1hot, logits=logits_theta)
	loss_theta = tf.reduce_mean(out_theta)

	# loss on phi
	logits_phi = conv2d(layers2d[-1], 13, 1, padding='SAME')
	out_phi = tf.nn.softmax_cross_entropy_with_logits_v2(labels=phi1hot, logits=logits_phi)
	loss_phi = tf.reduce_mean(out_phi)

	# symmetrize
	layers2d.append(0.5 * (layers2d[-1] + tf.transpose(layers2d[-1], perm=[0,2,1,3])))

	# loss on distances
	logits_dist = conv2d(layers2d[-1], nbins+1, 1, padding='SAME')
	out_dist = tf.nn.softmax_cross_entropy_with_logits_v2(labels=dist1hot, logits=logits_dist)
	loss_dist = tf.reduce_mean(out_dist)
	prob_dist = tf.nn.softmax(logits_dist)
   
	# loss on beta-strand pairings
	logits_bb = conv2d(layers2d[-1], 3, 1, padding='SAME')
	out_bb = tf.nn.softmax_cross_entropy_with_logits_v2(labels=bb1hot, logits=logits_bb)
	loss_bb = tf.reduce_mean(out_bb)

	# loss on omega
	logits_omega = conv2d(layers2d[-1], 25, 1, padding='SAME')
	out_omega = tf.nn.softmax_cross_entropy_with_logits_v2(labels=omega1hot, logits=logits_omega)
	loss_omega = tf.reduce_mean(out_omega)


	#
	# L2 penalty
	#
	vars = tf.trainable_variables()
	lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
					   in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
	
	
	#
	# optimizer
	#
	opt = tf.train.AdamOptimizer(learning_rate=lr)  #optimization

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	
	# training op
	with tf.control_dependencies(update_ops):
		train_op = opt.minimize(loss_dist + loss_bb + loss_theta + loss_phi + loss_omega + lossL2)


	total_parameters=np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
	print("tot. params: " + str(total_parameters))


	init_op = tf.group(tf.global_variables_initializer(), 
					   tf.local_variables_initializer())

	saver = tf.train.Saver()

	

	with tf.Session(config=config) as sess:

		sess.run(init_op)

		val_loss_best = 999.9
		
		out_name = method + '_' + "Test" + str(len(test))
		with open("/dl/suhong/project/TemptrRosetta/output/acc/%s.result"%(out_name), 'wt') as fid:
		
			for epoch in range(nb_epochs):

				# training
				random.shuffle(train) #rearrange train list
				train_losses = []
				for item in train:
					_, ldist, ltheta, lphi, lomega = sess.run([train_op, loss_dist, loss_theta, loss_phi, loss_omega],
															  feed_dict=get_fdict(item, 1))
					train_losses.append([ldist,ltheta,lphi,lomega])

				# testing
				val_losses = []
				vacc_dist = []
				random.shuffle(test)
				for item in test:
					ldist, ltheta, lphi, lomega, pdist, neff_ = sess.run([loss_dist, loss_theta, loss_phi, loss_omega, prob_dist, neff],
																		 feed_dict = get_fdict(item, 0))
					item['pdist'] = pdist
					item['neff'] = neff_
					val_losses.append([ldist,ltheta,lphi,lomega])
					start = item['start']
					stop = item['stop']
					vacc_dist.append(cont_acc(item['pdist'], item['dist'][start:stop,start:stop]))

					# output each test result in a new file
					acc = cont_acc(item['pdist'], item['dist'][start:stop,start:stop])
					out_str = item['label'] + '\t' + str('{:.5f}'.format(acc))
					fid.writelines(out_str+"\n")
				fid.writelines('epoch=%s\tval_acc=%s'%(str(epoch),str('{:.5f}'.format(np.average(vacc_dist))))+"\n")

				train_losses = np.array(train_losses)
				val_losses = np.array(val_losses)
				print("epoch {:3d} | train_losses {:.5f} {:.5f} {:.5f} {:.5f} | val_losses {:.5f} {:.5f} {:.5f} {:.5f} | val_acc {:.5f}".
					  format(epoch, 
							 np.average(train_losses[:,0]), np.average(train_losses[:,1]), 
							 np.average(train_losses[:,2]), np.average(train_losses[:,3]), 
							 np.average(val_losses[:,0]), np.average(val_losses[:,1]), 
							 np.average(val_losses[:,2]), np.average(val_losses[:,3]), 
							 np.average(vacc_dist)))
				sys.stdout.flush()
				saver.save(sess, CHK)
'''
'''
