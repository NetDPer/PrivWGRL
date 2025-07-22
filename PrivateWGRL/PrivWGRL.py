import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
from sklearn import metrics
from sklearn.externals import joblib
from rdp_accountant import compute_rdp, get_privacy_spent
import time
import random
import functions
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--batch_size', default=5000)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--sigma', default=5)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--epsilon', default=6)
parser.add_argument('--RDP', default=True)
parser.add_argument('--clip_value', default=0.1)
parser.add_argument('--neg_samp_num', default=5)
parser.add_argument('--run_time', default=0)
parser.add_argument('--consider_weight', default=True)
parser.add_argument('--layer_num', default=2)

class PrivWGraLearn_Model:
    def __init__(self, num_of_nodes):
        with tf.compat.v1.variable_scope('forward_pass'):
            tf.compat.v1.disable_eager_execution()
            self.w_i = tf.compat.v1.placeholder(name='w_i', dtype=tf.int32, shape=[None])
            self.h_j = tf.compat.v1.placeholder(name='h_j', dtype=tf.int32, shape=[None])
            self.PMI_values = tf.compat.v1.placeholder(name='point_mutual_info', dtype=tf.float32, shape=None)
            self.input_W = tf.compat.v1.get_variable('W_embedding_1', [num_of_nodes, args.embedding_dim],
                                               initializer=tf.keras.initializers.GlorotNormal())

            self.w_i_embedding = tf.matmul(tf.one_hot(self.w_i, depth=num_of_nodes), self.input_W)
            self.h_j_embedding = tf.matmul(tf.one_hot(self.h_j, depth=num_of_nodes), self.input_W)

            n_emb = args.embedding_dim
            n_hidden_prox = args.embedding_dim
            n_latent = args.embedding_dim

            # proximity representation
            self.W_hp_1 = tf.compat.v1.get_variable(name="W_hidden_prox_1", dtype=tf.float32, shape=[n_emb, n_hidden_prox])
            b_hp_1 = tf.compat.v1.get_variable(name="b_hidden_prox_1", dtype=tf.float32, shape=[n_hidden_prox])
            self.w_i_embedding = tf.nn.sigmoid(tf.add(tf.matmul(self.w_i_embedding, self.W_hp_1), b_hp_1))
            self.h_j_embedding = tf.nn.sigmoid(tf.add(tf.matmul(self.h_j_embedding, self.W_hp_1), b_hp_1))

            # proximity representation
            self.W_hp_2 = tf.compat.v1.get_variable(name="W_hidden_prox_2", dtype=tf.float32, shape=[n_emb, n_hidden_prox])
            b_hp_2 = tf.compat.v1.get_variable(name="b_hidden_prox_2", dtype=tf.float32, shape=[n_hidden_prox])
            self.w_i_embedding = tf.nn.sigmoid(tf.add(tf.matmul(self.w_i_embedding, self.W_hp_2), b_hp_2))
            self.h_j_embedding = tf.nn.sigmoid(tf.add(tf.matmul(self.h_j_embedding, self.W_hp_2), b_hp_2))

            self.pos_score = tf.reduce_sum(tf.multiply(self.w_i_embedding, self.h_j_embedding), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.pos_score - self.PMI_values))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr)
            self.params = [v for v in tf.compat.v1.trainable_variables() if 'forward_pass' in v.name]

            if args.RDP:
                self.var_list = self.params
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.var_list)
                for i, (g, v) in enumerate(self.grads_and_vars):
                    if g is not None and v is not None:
                        g = tf.clip_by_norm(g, args.clip_value)
                        stddev = args.sigma * args.clip_value / args.batch_size
                        g = g + tf.compat.v1.random_normal(tf.shape(g), stddev=stddev)
                        self.grads_and_vars[i] = (g, v)
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
            else:
                self.train_op = self.optimizer.minimize(self.loss)

class trainModel:
    def __init__(self, inf_display, graph, original_graph=None, test_pos=None, test_neg=None, test_ratio=None, node_label=None):
        self.inf_display = inf_display
        self.node_label = node_label
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.test_ration = test_ratio
        self.graph = graph
        self.original_graph = original_graph
        self.num_of_edges = self.graph.number_of_edges()
        self.num_of_nodes = graph.number_of_nodes()
        self.model = PrivWGraLearn_Model(self.num_of_nodes)

    def train(self, test_task=None, test_ratios=None, output_filename=None):
        global mse_max, mse_min, mse_mean, pearson_max, pearson_min, pearson_mean
        total_IterVals = []
        best_auc = []
        finalIter_auc = []

        mse_values = []
        pearson_values = []
        time_values = []

        for indep_run_time in range(args.indep_run_times):
            # start time
            start_time = time.time()
            IterVals = []
            found = False

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())  # note that this initilization's location
                flag_auc = 0
                orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                # orders = np.arange(2, 32, 0.1)
                rdp = np.zeros_like(orders, dtype=float)

                for each_epoch in range(args.n_epoch):
                    # print(indep_run_time, each_epoch)
                    w_i, h_j, weight_values, PMI_values = [], [], [], []
                    selected_edges_idx = np.random.choice(self.graph.number_of_edges(), size=args.batch_size, replace=False)  # 随机选择 128 条索引)
                    # get the list of edges
                    all_edges = list(self.graph.edges())
                    selected_edges = [all_edges[i] for i in selected_edges_idx]

                    for edge in selected_edges:
                        source, target = edge
                        w_i.append(source)
                        h_j.append(target)
                        weight_value = self.graph.get_edge_data(source, target).get('weight')
                        source_degree = self.graph.degree(source)
                        target_degree = self.graph.degree(target)
                        pmi = weight_value / source_degree / target_degree / args.neg_samp_num
                        PMI_values.append(np.log(pmi))
                        # PMI_values.append(pmi)

                    # PMI_values = [max(val, 0) for val in PMI_values]
                    feed_dict = {self.model.w_i: w_i, self.model.h_j: h_j, self.model.PMI_values: PMI_values}

                    _, loss, W_emb = sess.run([self.model.train_op, self.model.loss, self.model.input_W], feed_dict=feed_dict)

                    print('Indep_run_time', indep_run_time, 'Epoch', each_epoch)

                    if args.RDP:
                        sampling_prob = args.batch_size / self.graph.number_of_edges()
                        steps = each_epoch + 1
                        # different rdp computation is available in rdp_accountant
                        rdp = compute_rdp(q=sampling_prob, noise_multiplier=args.sigma, steps=steps, orders=orders)

                        _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=args.epsilon)

                        if _delta > args.delta:
                            print('jump out')
                            found = True
                            break

                    if found:
                        break

                if test_task == 'lwp':
                    # Convert W_emb to a sparse matrix (if it is dense, this will make it sparse)
                    W_emb_sparse = csr_matrix(W_emb)
                    # Perform the dot product (this will be sparse)
                    embedding_mat_sparse = W_emb_sparse.dot(W_emb_sparse.T)
                    # If you need to work with the dense matrix, convert it to dense (this step should be avoided if possible)
                    embedding_mat = embedding_mat_sparse.toarray()
                    # embedding_mat = np.dot(W_emb, W_emb.T)
                    mse_value = CalcAUC(self.original_graph, embedding_mat, test_pos, test_neg)

                    print('Indep_run_time', indep_run_time, 'Epoch', each_epoch, 'MSE_Value', mse_value)

                if test_task == 'StrucEqu':
                    A = nx.to_numpy_matrix(trainGraph)
                    A = np.array(A)
                    W_emb[W_emb < 0] = 0
                    pearson_vals = functions.structural_equivalence(A, W_emb)
                    pearson_val = pearson_vals[0]
                    print('Indep_run_time', indep_run_time, 'Epoch', each_epoch)

                # end time
                end_time = time.time()
                run_time = (end_time - start_time) / 60  # save time on min
                time_values.append(run_time)

                if test_task == 'lwp':
                    mse_values.append(mse_value)

                if test_task == 'StrucEqu':
                    pearson_values.append(pearson_val)

        if test_task == 'lwp':
            # computer max、min and mean
            mse_max = np.max(mse_values)
            mse_min = np.min(mse_values)
            mse_mean = np.mean(mse_values)

        if test_task == 'StrucEqu':
            # computer max、min and mean
            pearson_max = np.max(pearson_values)
            pearson_min = np.min(pearson_values)
            pearson_mean = np.mean(pearson_values)

        time_max = np.max(time_values)
        time_min = np.min(time_values)
        time_mean = np.mean(time_values)

        mark_time = str(time.time()).split(".")[0]
        output_final_name = output_filename + '_' + mark_time

        # save the results to a txt file
        with open(output_final_name + '.txt', 'w') as f:
            if test_task == 'lwp':
                f.write(f"mse_values = {mse_values}\n")
                f.write(f"max: {mse_max:.6f}\n")  # keep 6 decimal places
                f.write(f"min: {mse_min:.6f}\n")
                f.write(f"mean: {mse_mean:.6f}\n\n")

            if test_task == 'StrucEqu':
                f.write(f"pearson_values = {pearson_values}\n")
                f.write(f"max: {pearson_max:.6f}\n")  # keep 6 decimal places
                f.write(f"min: {pearson_min:.6f}\n")
                f.write(f"mean: {pearson_mean:.6f}\n\n")

            # save time_values
            f.write(f"time_values = {time_values}\n")
            f.write(f"max: {time_max:.6f}\n")  # keep 6 decimal places
            f.write(f"min: {time_min:.6f}\n")
            f.write(f"mean: {time_mean:.6f}\n")

def CalcAUC(graph, sim, test_pos, test_neg):
    u_list = test_pos[0]
    v_list = test_pos[1]

    pos_label_list = []
    for u, v in zip(u_list, v_list):
        u_v_weight = graph.get_edge_data(u, v)['weight']

        pos_label_list.append(u_v_weight)

    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    data_min = np.min(scores)
    data_max = np.max(scores)
    scores = (scores - data_min) / (data_max - data_min)

    labels = np.hstack([pos_label_list, np.zeros(len(neg_scores))])
    data_min = np.min(labels)
    data_max = np.max(labels)
    labels = (labels - data_min) / (data_max - data_min)

    mse_value = metrics.mean_squared_error(labels, scores)

    return mse_value

def loadGraphFromEdgeListTxt(file_name, directed=True):
    with open(file_name, 'r') as f:
        # n_nodes = f.readline()
        # f.readline() # Discard the number of edges
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for line in f:
            edge = line.strip().split()
            if len(edge) == 3:
                if args.consider_weight:
                    w = float(edge[2])
                else:
                    w = 1
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=w)

    return G

if __name__ == '__main__':
    test_task = 'StrucEqu'  # lwp or StrucEqu
    set_algo_name = 'PrivWGRL-ILF_varyEps'

    parser.add_argument('--n_epoch', default=5)  # all datasets are set to 5
    parser.add_argument('--indep_run_times', default=5)  # done
    args = parser.parse_args()  # parameter

    if test_task == 'lwp':
        set_dataset_names = ['Reality-call']

        set_split_name = 'train0.8_test0.2'
        set_nepoch_name = 'nepoch' + str(args.n_epoch)
        set_emb_dim = 'dim' + str(args.embedding_dim)
        set_layer_num = 'layerNum' + str(args.layer_num)
        set_learning_rate = 'step' + str(args.lr)
        set_isWeight = 'Weight' + str(args.consider_weight)
        set_clip_value = 'clip' + str(args.clip_value)

        for set_dataset_name in set_dataset_names:
            if set_dataset_name is 'Reality-call':
                set_batch_value = 2000
            else:
                set_batch_value = 10000

            args.batch_size = set_batch_value
            set_batch = 'batch' + str(args.batch_size)

            set_eps_values = [1]
            for each_eps_value in set_eps_values:
                args.epsilon = each_eps_value
                set_eps_value = 'eps' + str(args.epsilon)

                tf.compat.v1.reset_default_graph()
                oriGraph_filename = '../data/' + set_dataset_name + '/train_1'
                train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

                output_filename = test_task + '_' + set_algo_name + '_' + set_dataset_name + '_' + set_isWeight + '_' \
                                  + set_split_name + '_' + set_emb_dim + '_' + set_nepoch_name + '_' \
                                  + set_eps_value + '_' + set_learning_rate + '_' + set_batch + '_' \
                                  + set_clip_value + '_' + set_layer_num

                # Load graph
                trainGraph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)

                print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))

                original_graph = trainGraph

                trainGraph = nx.adjacency_matrix(trainGraph)

                train_pos = joblib.load(train_filename + 'train_pos.pkl')
                train_neg = joblib.load(train_filename + 'train_neg.pkl')
                test_pos = joblib.load(train_filename + 'test_pos.pkl')
                test_neg = joblib.load(train_filename + 'test_neg.pkl')
                # train_pos, train_neg, test_pos, test_neg = sample_neg(trainGraph, test_ratio=0.2, max_train_num=100000)

                trainGraph = trainGraph.copy()  # the observed network
                trainGraph[test_pos[0], test_pos[1]] = 0  # mask test links
                trainGraph[test_pos[1], test_pos[0]] = 0  # mask test links
                trainGraph.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

                row, col = train_neg
                trainGraph = trainGraph.copy()
                trainGraph[row, col] = 1  # inject negative train
                trainGraph[col, row] = 1  # inject negative train
                trainGraph = nx.from_scipy_sparse_matrix(trainGraph)

                inf_display = [test_task, set_dataset_name]
                tm = trainModel(inf_display, trainGraph, original_graph=original_graph, test_pos=test_pos, test_neg=test_neg)
                tm.train(test_task=test_task, test_ratios=[0], output_filename=output_filename)

    if test_task == 'StrucEqu':
        set_dataset_names = ['Reality-call']

        set_split_name = 'train0.8_test0.2'
        set_nepoch_name = 'nepoch' + str(args.n_epoch)
        set_emb_dim = 'dim' + str(args.embedding_dim)
        set_layer_num = 'layerNum' + str(args.layer_num)
        set_learning_rate = 'step' + str(args.lr)
        set_isWeight = 'Weight' + str(args.consider_weight)
        set_clip_value = 'clip' + str(args.clip_value)

        for set_dataset_name in set_dataset_names:
            if set_dataset_name is 'Reality-call':
                set_batch_value = 2000
            else:
                set_batch_value = 10000

            args.batch_size = set_batch_value
            set_batch = 'batch' + str(args.batch_size)

            set_eps_values = [1]
            for each_eps_value in set_eps_values:
                args.epsilon = each_eps_value
                set_eps_value = 'eps' + str(args.epsilon)

                tf.compat.v1.reset_default_graph()
                oriGraph_filename = '../data/' + set_dataset_name + '/train_1'
                train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

                output_filename = test_task + '_' + set_algo_name + '_' + set_dataset_name + '_' + set_isWeight + '_' \
                                  + set_split_name + '_' + set_emb_dim + '_' + set_nepoch_name + '_' \
                                  + set_eps_value + '_' + set_learning_rate + '_' + set_batch + '_' \
                                  + set_clip_value + '_' + set_layer_num

                # Load graph
                trainGraph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)
                original_graph = trainGraph

                print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
                inf_display = [test_task, set_dataset_name]
                tm = trainModel(inf_display, trainGraph, original_graph=original_graph)
                tm.train(test_task=test_task, test_ratios=[0], output_filename=output_filename)


