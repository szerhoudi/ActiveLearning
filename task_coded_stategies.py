"""
Active Learning - choosing queries strategies
(UncertaintySampling (Max Margin) - CMB Sampling : Combination of active learning algorithms 
    (distance-based (DIST), diversity-based (DIV)))

"""

import copy
import time
import codecs

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from strategies.strategies import USampling, CMBSampling
from sklearn.metrics import accuracy_score
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SVM
from libact.query_strategies import UncertaintySampling, RandomSampling, QueryByCommittee, HintSVM


def openfile_txt(filepath):
    with open(filepath, 'r', encoding='utf16') as f:
        file = f.read().split('"\n[')
    return file


def simulate_w4v(tweet_id):
    element_id = ids_list.index(tweet_id)
    vectorized = vectors_list[element_id]
    return vectorized


def get_vectors_list(filepath):
    vectors_list_x, ids_list_x = [], []
    with open(filepath, 'r', encoding='utf16') as f:
        file = f.read().split('"\n[')
        for line in file:
            parts = line.replace('\n', '').replace('    ', ' ').replace('  ', ' ').replace('  ', ' ').split(";")
            vectors_list_x.append(parts[0])
            ids_list_x.append(parts[1].replace(' ', ''))
    return vectors_list_x, ids_list_x


def define_label(tweet_id):
    with open(pos_filepath, 'r', encoding='utf16') as f:
        next(f)
        for line in f.readlines():
            parts = line.split(";")
            tweets = parts[0].replace('"', '')
            if tweet_id in tweets:
                label = 1
                break
            else:
                label = 0
    return label


def define_tweet_by_id(line_id):
    with open(csv_filepath, 'r', encoding='utf16') as fp:
        for i, line in enumerate(fp):
            if i == line_id:
                parts = line.split(";")
                tweet = parts[2]
            elif i > line_id:
                break
    return tweet


def randomize(X, y):
    permutation = np.random.permutation(y.shape[0])
    X2 = X[permutation]
    y2 = y[permutation]
    return X2, y2


def build_dataset(filepath):
    target, data = [], []
    with open(filepath, 'r', encoding='utf16') as f:
        next(f)
        for line in f.readlines()[:]:
            parts = line.split(";")
            z = np.array(define_label(parts[0]))
            target.append(z)
            x = np.fromstring(simulate_w4v(parts[0]).replace(']', '').replace('[', '').replace('  ', ' '), sep=' ')
            data.append(x)
            ''' Test with random dataset
            target = np.random.randint(2, size=(500, ))
            data = np.random.uniform(low=-10.5, high=10.3, size=(500, 100))'''
    target = np.asarray(target)
    data = np.asarray(data)
    return target, data


def simulate_human_decision(line_id):
    with open(csv_filepath, 'r', encoding='utf16') as fp:
        for i, line in enumerate(fp):
            if i == line_id:
                parts = line.split(";")
                tweet_id = parts[0]
                label = define_label(tweet_id)
            elif i > line_id:
                break
    return label


def split_train_test(csv_filepath):
    target = build_dataset(csv_filepath)
    n_labeled = 10

    X = target[1]
    y = target[0]
    print(np.shape(X))
    print(np.shape(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    while (np.sum(y_train[:n_labeled]) < 2):
        X_rand, y_rand = randomize(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_rand, y_rand, test_size=0.2, stratify=y_rand)

    print(np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))

    trn_ds = Dataset(X_train, np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)

    return trn_ds, tst_ds


def main():
    global pos_filepath, dataset_filepath, csv_filepath, vectors_list, ids_list
    dataset_filepath = "/Users/dndesign/Desktop/active_learning/vecteurs_et_infos/vectors_2015.txt"
    csv_filepath = "/Users/dndesign/Desktop/active_learning/donnees/corpus_2015_id-time-text.csv"
    pos_filepath = "/Users/dndesign/Desktop/active_learning/donnees/oriane_pos_id-time-text.csv"
    vectors_list, ids_list = get_vectors_list(dataset_filepath)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    text_file = codecs.open("task_" + str(timestr) + ".txt", "w", "utf-8")

    print("Loading data...")
    text_file.write("Loading data...\n")
    # Open this file
    t0 = time.time()
    file = openfile_txt(dataset_filepath)
    num_lines = sum(1 for line in file)
    print("Treating " + str(num_lines) + " entries...")
    text_file.write("Treating : %s entries...\n" % str(num_lines))

    # Number of queries to ask human to label
    quota = 10
    E_out1, E_out2, E_out3, E_out4, E_out6, E_out7 = [], [], [], [], [], []
    trn_ds, tst_ds = split_train_test(csv_filepath)

    model = SVM(kernel='linear')
    # model = LogisticRegression()

    ''' UncertaintySampling (Least Confident)
     
        UncertaintySampling : it queries the instances about which 
        it is least certain how to label
        
        Least Confident : it queries the instance whose posterior 
        probability of being positive is nearest 0.5
    '''
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression(C=.01))
    model.train(trn_ds)
    E_out1 = np.append(E_out1, 1 - model.score(tst_ds))

    ''' UncertaintySampling (Max Margin) 

    '''
    trn_ds2 = copy.deepcopy(trn_ds)
    qs2 = USampling(trn_ds2, method='mm', model=SVM(kernel='linear'))
    model.train(trn_ds2)
    E_out2 = np.append(E_out2, 1 - model.score(tst_ds))

    ''' CMB Sampling   
        Combination of active learning algorithms (distance-based (DIST), diversity-based (DIV)) 
    '''
    trn_ds3 = copy.deepcopy(trn_ds)
    qs3 = CMBSampling(trn_ds3, model=SVM(kernel='linear'))
    model.train(trn_ds3)
    E_out3 = np.append(E_out3, 1 - model.score(tst_ds))

    ''' Random Sampling   
        Random : it chooses randomly a query
    '''
    trn_ds4 = copy.deepcopy(trn_ds)
    qs4 = RandomSampling(trn_ds4, random_state=1126)
    model.train(trn_ds4)
    E_out4 = np.append(E_out4, 1 - model.score(tst_ds))

    ''' QueryByCommittee (Vote Entropy)
    
        QueryByCommittee : it keeps a committee of classifiers and queries 
        the instance that the committee members disagree, it  also examines 
        unlabeled examples and selects only those that are most informative 
        for labeling
        
        Vote Entropy : a way of measuring disagreement 
        
        Disadvantage : it does not consider the committee members’ class 
        distributions. It also misses some informative unlabeled examples 
        to label 
    '''
    trn_ds6 = copy.deepcopy(trn_ds)
    qs6 = QueryByCommittee(trn_ds6, disagreement='vote',
                              models=[LogisticRegression(C=1.0),
                                      LogisticRegression(C=0.01),
                                      LogisticRegression(C=100)],
                              random_state=1126)
    model.train(trn_ds6)
    E_out6 = np.append(E_out6, 1 - model.score(tst_ds))

    ''' QueryByCommittee (Kullback-Leibler Divergence)
    
            QueryByCommittee : it examines unlabeled examples and selects only 
            those that are most informative for labeling
            
            Disadvantage :  it misses some examples on which committee members 
            disagree
    '''
    trn_ds7 = copy.deepcopy(trn_ds)
    qs7 = QueryByCommittee(trn_ds7, disagreement='kl_divergence',
                                  models=[LogisticRegression(C=1.0),
                                          LogisticRegression(C=0.01),
                                          LogisticRegression(C=100)],
                                  random_state=1126)
    model.train(trn_ds7)
    E_out7 = np.append(E_out7, 1 - model.score(tst_ds))

    with sns.axes_style("darkgrid"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    query_num = np.arange(0, 1)
    p1, = ax.plot(query_num, E_out1, 'red')
    p2, = ax.plot(query_num, E_out2, 'blue')
    p3, = ax.plot(query_num, E_out3, 'green')
    p4, = ax.plot(query_num, E_out4, 'orange')
    p6, = ax.plot(query_num, E_out6, 'black')
    p7, = ax.plot(query_num, E_out7, 'purple')
    plt.legend(('Least Confident', 'Max Margin', 'Distance Diversity CMB', 'Random Sampling', 'Vote Entropy', 'KL Divergence'), loc=1)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Queries')
    plt.title('Active Learning - Query choice strategies')
    plt.ylim([0, 1])
    plt.show(block=False)

    for i in range(quota):
        print("\n#################################################")
        print("Query number " + str(i) + " : ")
        print("#################################################\n")
        text_file.write("\n#################################################\n")
        text_file.write("Query number %s : " % str(i))
        text_file.write("\n#################################################\n")

        ask_id = qs.make_query()
        print("\033[4mUsing Uncertainty Sampling (Least confident) :\033[0m")
        print("Tweet :" + define_tweet_by_id(ask_id), end='', flush=True)
        print("Simulating human response : " + str(simulate_human_decision(ask_id)) + " \n")
        text_file.write("Using Uncertainty Sampling (Least confident) :\n")
        text_file.write("Tweet : %s \n" % str(define_tweet_by_id(ask_id)))
        text_file.write("Simulating human response : %s \n\n" % str(simulate_human_decision(ask_id)))
        trn_ds.update(ask_id, simulate_human_decision(ask_id))
        model.train(trn_ds)
        E_out1 = np.append(E_out1, 1 - model.score(tst_ds))

        ask_id = qs2.make_query()
        print("\033[4mUsing Uncertainty Sampling (Max Margin) :\033[0m")
        print("Tweet :" + define_tweet_by_id(ask_id), end='', flush=True)
        print("Simulating human response : " + str(simulate_human_decision(ask_id)) + " \n")
        text_file.write("Using Uncertainty Sampling (Smallest Margin) :\n")
        text_file.write("Tweet : %s \n" % str(define_tweet_by_id(ask_id)))
        text_file.write("Simulating human response : %s \n\n" % str(simulate_human_decision(ask_id)))
        trn_ds2.update(ask_id, simulate_human_decision(ask_id))
        model.train(trn_ds2)
        E_out2 = np.append(E_out2, 1 - model.score(tst_ds))

        ask_id = qs3.make_query()
        print("\033[4mUsing CMB Distance-Diversity Sampling :\033[0m")
        print("Tweet :" + define_tweet_by_id(ask_id), end='', flush=True)
        print("Simulating human response : " + str(simulate_human_decision(ask_id)) + " \n")
        text_file.write("Using Uncertainty Sampling (Entropy) :\n")
        text_file.write("Tweet : %s \n" % str(define_tweet_by_id(ask_id)))
        text_file.write("Simulating human response : %s \n\n" % str(simulate_human_decision(ask_id)))
        trn_ds3.update(ask_id, simulate_human_decision(ask_id))
        model.train(trn_ds3)
        E_out3 = np.append(E_out3, 1 - model.score(tst_ds))

        ask_id = qs4.make_query()
        print("\033[4mUsing Random Sampling :\033[0m")
        print("Tweet :" + define_tweet_by_id(ask_id), end='', flush=True)
        print("Simulating human response : " + str(simulate_human_decision(ask_id)) + " \n")
        text_file.write("Using Random Sampling :\n")
        text_file.write("Tweet : %s \n" % str(define_tweet_by_id(ask_id)))
        text_file.write("Simulating human response : %s \n\n" % str(simulate_human_decision(ask_id)))
        trn_ds4.update(ask_id, simulate_human_decision(ask_id))
        model.train(trn_ds4)
        E_out4 = np.append(E_out4, 1 - model.score(tst_ds))

        ask_id = qs6.make_query()
        print("\033[4mUsing QueryByCommittee (Vote Entropy) :\033[0m")
        print("Tweet :" + define_tweet_by_id(ask_id), end='', flush=True)
        print("Simulating human response : " + str(simulate_human_decision(ask_id)) + " \n")
        text_file.write("Using QueryByCommittee (Vote Entropy) :\n")
        text_file.write("Tweet : %s \n" % str(define_tweet_by_id(ask_id)))
        text_file.write("Simulating human response : %s \n\n" % str(simulate_human_decision(ask_id)))
        trn_ds6.update(ask_id, simulate_human_decision(ask_id))
        model.train(trn_ds6)
        E_out6 = np.append(E_out6, 1 - model.score(tst_ds))

        ask_id = qs7.make_query()
        print("\033[4mUsing QueryByCommittee (KL Divergence) :\033[0m")
        print("Tweet :" + define_tweet_by_id(ask_id), end='', flush=True)
        print("Simulating human response : " + str(simulate_human_decision(ask_id)) + " \n")
        text_file.write("Using QueryByCommittee (KL Divergence) :\n")
        text_file.write("Tweet : %s \n" % str(define_tweet_by_id(ask_id)))
        text_file.write("Simulating human response : %s \n\n" % str(simulate_human_decision(ask_id)))
        trn_ds7.update(ask_id, simulate_human_decision(ask_id))
        model.train(trn_ds7)
        E_out7 = np.append(E_out7, 1 - model.score(tst_ds))

        ax.set_xlim((0, i + 1))
        ax.set_ylim((0, max(max(E_out1), max(E_out2), max(E_out3), max(E_out4), max(E_out6), max(E_out7)) + 0.2))
        query_num = np.arange(0, i + 2)
        p1.set_xdata(query_num)
        p1.set_ydata(E_out1)
        p2.set_xdata(query_num)
        p2.set_ydata(E_out2)
        p3.set_xdata(query_num)
        p3.set_ydata(E_out3)
        p4.set_xdata(query_num)
        p4.set_ydata(E_out4)
        p6.set_xdata(query_num)
        p6.set_ydata(E_out6)
        p7.set_xdata(query_num)
        p7.set_ydata(E_out7)

        plt.draw()

    t2 = time.time()
    time_total = t2 - t0
    print("\n\n\n#################################################\n")
    print("Execution time : %fs \n\n" % time_total)
    text_file.write("\n\n\n#################################################\n")
    text_file.write("Execution time : %fs \n" % time_total)
    text_file.close()
    input("Press any key to save the plot...")
    plt.savefig('task_' + str(timestr) + '.png')

    print("Done")


if __name__ == '__main__':
    main()
