import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
from sklearn import metrics
from sklearn.externals import joblib
from rdp_accountant import compute_rdp, get_privacy_spent

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--batch_size', default=10000)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--sigma', default=5)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--epsilon', default=6)
parser.add_argument('--RDP', default=True)
parser.add_argument('--clip_value', default=0.1)
parser.add_argument('--neg_samp_num', default=5)
parser.add_argument('--run_time', default=0)
parser.add_argument('--progressive_perturb', default=True)
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
        for indep_run_time in range(args.indep_run_times):
            found = False
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                rdp = np.zeros_like(orders, dtype=float)

                for each_epoch in range(args.n_epoch):
                    w_i, h_j, weight_values, PMI_values = [], [], [], []
                    selected_edges_idx = np.random.choice(self.graph.number_of_edges(), size=args.batch_size, replace=False)
                    all_edges = list(self.graph.edges())
                    selected_edges = [all_edges[i] for i in selected_edges_idx]

                    for edge in selected_edges:
                        source, target = edge
                        w_i.append(source)
                        h_j.append(target)
                        weight_value = self.graph.get_edge_data(source, target).get('weight')
                        source_degree = self.graph.degree(source)
                        target_degree = self.graph.degree(target)
                        pmi = weight_value / source_degree / target_degree
                        PMI_values.append(np.log(pmi))

                    feed_dict = {self.model.w_i: w_i, self.model.h_j: h_j, self.model.PMI_values: PMI_values}

                    _, loss, W_emb = sess.run([self.model.train_op, self.model.loss, self.model.input_W], feed_dict=feed_dict)

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

                    print('Indep_run_time', indep_run_time, 'Epoch', each_epoch)

def loadGraphFromEdgeListTxt(file_name, directed=True):
    with open(file_name, 'r') as f:
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
    test_task = 'lwp'  # link weight prediction
    set_algo_name = 'PrivWGRL'

    parser.add_argument('--n_epoch', default=300)
    parser.add_argument('--indep_run_times', default=5)
    args = parser.parse_args()  # parameters

    set_dataset_names = ['lp_w_reality-call', 'lp_w_contacts-dublin', 'lp_w_digg-reply',
                         'lp_w_graph_Enron', 'lp_w_dblp', 'lp_w_wiki']

    set_split_name = 'train0.8_test0.2'
    for set_dataset_name in set_dataset_names:
        if set_dataset_name is 'lp_w_reality-call':
            set_batch_value = 2000
        else:
            set_batch_value = 10000

        args.batch_size = set_batch_value

        set_eps_values = [6]
        for each_eps_value in set_eps_values:
            args.epsilon = each_eps_value

            tf.compat.v1.reset_default_graph()
            oriGraph_filename = '../data/' + set_dataset_name + '/train_1'
            train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

            # Load graph
            trainGraph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)
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

            print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
            inf_display = [test_task, set_dataset_name]
            tm = trainModel(inf_display, trainGraph, original_graph=original_graph, test_pos=test_pos, test_neg=test_neg)
            tm.train(test_task=test_task, test_ratios=[0], output_filename=set_algo_name)



