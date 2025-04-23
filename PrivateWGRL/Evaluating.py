import numpy as np
import argparse
from sklearn import metrics
import networkx as nx
from sklearn.externals import joblib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--consider_weight', default=True)
args = parser.parse_args()  # parameters

def CalcLWP(graph, sim, test_pos, test_neg):
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
    mae_value = metrics.mean_absolute_error(labels, scores)

    return mse_value, mae_value

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
    test_task = 'lwp'

    if test_task == 'lwp':
        set_dataset_names = ['lp_w_reality-call']

        for set_dataset_name in set_dataset_names:
            set_split_name = 'train0.8_test0.2'
            oriGraph_filename = '../data/' + set_dataset_name + '/train_1'
            train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

            # Reset variables before each run
            trainGraph = None
            test_pos = None
            test_neg = None

            # Load graph
            trainGraph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)

            test_pos = joblib.load(train_filename + 'test_pos.pkl')
            test_neg = joblib.load(train_filename + 'test_neg.pkl')

            # Path of the folder
            name1 = set_dataset_names[0]
            folder_path = Path(name1)

            # Get all files in the folder
            # Get all files in the folder and remove the .npz suffix
            file_names_without_npz = [file.stem for file in folder_path.iterdir() if
                                      file.is_file() and file.suffix == '.npz']

            for each_file in file_names_without_npz:
                # Given values
                mse_values = []
                mae_values = []
                time_values = []

                data = np.load(name1 + '/' + each_file + '.npz')

                # Access the saved arrays
                W_embs = data['W_embs']  # get W_embs
                a_val = W_embs.shape[0]
                times = data['times']  # get times

                for t in range(a_val):
                    W_emb = W_embs[t]
                    time = times[t]
                    embedding_mat = np.dot(W_emb, W_emb.T)
                    mse_value, mae_value = CalcLWP(trainGraph, embedding_mat, test_pos, test_neg)

                    mse_values.append(mse_value)
                    mae_values.append(mae_value)
                    time_values.append(time)

                # calculate max min mean values
                mse_max = np.max(mse_values)
                mse_min = np.min(mse_values)
                mse_mean = np.mean(mse_values)

                mae_max = np.max(mae_values)
                mae_min = np.min(mae_values)
                mae_mean = np.mean(mae_values)

                time_max = np.max(time_values)
                time_min = np.min(time_values)
                time_mean = np.mean(time_values)

                # save results
                with open(each_file + '.txt', 'w') as f:
                    # save mse_values
                    f.write(f"mse_values = {mse_values}\n")
                    f.write(f"max: {mse_max:.6f}\n")
                    f.write(f"min: {mse_min:.6f}\n")
                    f.write(f"mean: {mse_mean:.6f}\n\n")

                    # save mae_values
                    f.write(f"mae_values = {mae_values}\n")
                    f.write(f"max: {mae_max:.6f}\n")
                    f.write(f"min: {mae_min:.6f}\n")
                    f.write(f"mean: {mae_mean:.6f}\n\n")
