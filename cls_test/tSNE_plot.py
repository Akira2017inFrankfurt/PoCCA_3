import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from cls_eval_svm import *
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import argparse


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(aspect='equal')
    ax = plt.subplot(aspect='auto')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    # sc = plt.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    if num_classes == 10:
        colors = colors.squeeze()
    # for i in range(num_classes):
    #     # Position of each label at median of data points.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     # txt = ax.text(xtext, ytext, str(colors[i]), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    return f, ax, sc, txts


def fit_tsne(args):
    start = time.time()
    feats_train, labels_train, feats_test, labels_test = load_feats(args)
    feature_embedded = TSNE(n_components=2, random_state=0,
                            learning_rate='auto',
                            n_iter=2000,
                            early_exaggeration=6,
                            init='pca',
                            perplexity=10,
                            n_jobs=8
                            ).fit_transform(feats_test)

    print(f't-SNE Done! Time: {time.time() - start} seconds!')
    return feature_embedded, labels_test


def draw_tsne(x, y):
    f, ax, sc, _ = fashion_scatter(x, y)
    f.show()


def fit_dbscan(args):
    print('For eps: {} and min_samples: {}'.format(args.eps, args.min_samples))
    feats_train, labels_train, feats_test, labels_test = load_feats(args)
    x = feats_train
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(x)
    list_outlier = [i for i in clustering.labels_ if i < 0]
    print('labels number: ', clustering.labels_.max() + 1)
    print('outliers num: ', len(list_outlier))
    print(clustering.labels_[:200])


if __name__ == "__main__":
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    device = torch.device('cuda')
    model_choice = 1  # default
    dataset_choice = ['modelnet40', 'scan']
    batch_size = 128  # default for PointNet
    weight_file = r'/home/haruki/下载/SimAttention/scripts/weights/'

    name_list = [99]
    for name in name_list:
        weight_name = 'cls_dg_1206_cross_' + str(name)
        # weight_name = 'cls_dg_1125_multi_scale_' + str(name)
        if weight_name.startswith('cls_dg'):
            model_choice = 1
            batch_size = 32
        weight_path = os.path.join(weight_file, weight_name + '.pth')
        print('For this model[{}]'.format(weight_name))


    def parameters_init():
        parser = argparse.ArgumentParser(description='Point Cloud Classification t-SNE!')
        parser.add_argument('--model_choice', type=int, default=1, help='0 for PN, 1 for DGCNN')
        parser.add_argument('--weight_path', type=str, default=weight_path)
        parser.add_argument('--dataset', type=str, default='modelnet10', choices=['modelnet40', 'scan', 'modelnet10'])
        parser.add_argument('--batch_size', type=int, default=batch_size, help='ScanObjectNN use half')
        parser.add_argument('--num_points', type=int, default=1024)
        parser.add_argument('--eps', type=int, default=9.25)
        parser.add_argument('--min_samples', type=int, default=2)

        parsers = parser.parse_args()
        return parsers


    hyper_parameters = parameters_init()
    X, Y = fit_tsne(hyper_parameters)
    draw_tsne(X, Y)

    # palette = np.array(sns.color_palette("hls", 10))
    # print(palette)

