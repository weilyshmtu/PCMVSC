import os

os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import torch
import torch.optim as optim

import pcmvsc as M
import utilis as U



def train_model(X, model, optimizer, alpha, beta, epochs):
    l = []
    acc_array = []
    nmi_array = []
    f1_array = []
    ari_array = []
    C_best = []
    for epoch in range(epochs):
        x_v_rec, coef_v, coef_v_rec, coef_fusion = model(X)
        loss = model.loss(X, x_v_rec, coef_v, coef_v_rec, coef_fusion, alpha, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            C = coef_fusion.detach().to('cpu').numpy()

            acc, nmi, f1_macro, ari = U.get_cluster_results(C, labels.squeeze(1), n_classes)
            if len(acc_array) > 1 and acc > max(acc_array):
                C_best = C
            acc_array.append(acc)
            nmi_array.append(nmi)
            f1_array.append(f1_macro)
            ari_array.append(ari)
            print("epoch = %d, acc = %f, nmi = %f, f1 = %f, ari = %f, loss = %f" % (
                epoch, acc, nmi, f1_macro, ari, loss.item()))

        l.append(loss.item())
    return l, acc_array, nmi_array, f1_array, ari_array, C_best


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # load data
    current_directory = os.getcwd()
    datasets = ['3sources', 'WikipediaArticles', 'BBCsport', 'MSRC_v1', 'WebKB', 'Caltech101-20', 'Handwritten',
                '100Leaves']

    for data_name in datasets:
        X, labels, n_views, n_samples = U.load_data(data_name, current_directory)
        X = U.data_normalize_l2(X, n_views)

        n_classes = np.max(np.unique(labels))
        if np.min(np.unique(labels)) == 0:
            n_classes = n_classes + 1

        k_set = [3, 5, 8, 10, 12, 15, 18]
        temperature = 1
        for k in k_set:
            U.write_splitter(data_name)
            if isinstance(X[0], torch.Tensor):
                for i in range(n_views):
                    X[i] = X[i].detach().to('cpu').numpy()

            positive_adj_graphs = U.adj_graphs(X, n_samples, k, 'cosine')  #'euclidean' 'cosine'
            fused_adj_graph = U.fused_adj_graph(positive_adj_graphs, n_samples, n_views)
            adj_graph = torch.tensor(fused_adj_graph, dtype=torch.float32, device=device)

            if not isinstance(X, torch.Tensor):
                for i in range(n_views):
                    X[i] = torch.tensor(X[i], dtype=torch.float32, device=device)

            epochs = 500 + 1
            alpha_set = [0.001, 0.01, 0.1, 1, 10, 100]
            beta_set = [0.001, 0.01, 0.1, 1, 10, 100]
            for alpha in alpha_set:
                for beta in beta_set:
                    U.set_seed(0)
                    model = M.multi_view_contrastive_clustering(n_samples, n_views, temperature, adj_graph).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)

                    l, acc_array, nmi_array, f1_array, ari_array, C_best = train_model(X, model, optimizer, alpha, beta,
                                                                                       epochs)
                    U.write_best_results(data_name, temperature, k, alpha, beta, 0, acc_array, nmi_array, f1_array,
                                         ari_array, False)
                    a = 1
