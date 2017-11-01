from my_model_selectors import (SelectorCV, SelectorDIC, SelectorBIC, SelectorConstant)
import timeit
import numpy as np
import pandas as pd
from asl_data import AslDb
import warnings
from hmmlearn.hmm import GaussianHMM
import math
from matplotlib import (cm, pyplot as plt, mlab)
import timeit
from my_recognizer import recognize
from asl_utils import show_errors

def build_features(asl):

    # This list will store all the features
    features = dict()

    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()

    #Ground features
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
    features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
    features['ground'] = features_ground

    
    # Building norm features
    asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(df_means['right-x'])) / \
                        asl.df['speaker'].map(df_std['right-x'])
    asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(df_means['right-y'])) / \
                        asl.df['speaker'].map(df_std['right-y'])
    asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['speaker'].map(df_means['left-x'])) / \
                        asl.df['speaker'].map(df_std['left-x'])
    asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['speaker'].map(df_means['left-y'])) / \
                        asl.df['speaker'].map(df_std['left-y'])
    features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
    features['norm'] = features_norm

    # Building polar features
    asl.df['polar-rr'] = np.sqrt(np.power(asl.df['right-x'] - asl.df['nose-x'], 2) +
                                np.power(asl.df['right-y'] - asl.df['nose-y'], 2))
    asl.df['polar-rtheta'] = np.arctan2(asl.df['right-x'] - asl.df['nose-x'],
                                        asl.df['right-y'] - asl.df['nose-y'])
    asl.df['polar-lr'] = np.sqrt(np.power(asl.df['left-x'] - asl.df['nose-x'], 2) +
                                np.power(asl.df['left-y'] - asl.df['nose-y'], 2))
    asl.df['polar-ltheta'] = np.arctan2(asl.df['left-x'] - asl.df['nose-x'],
                                        asl.df['left-y'] - asl.df['nose-y'])
    features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
    features['polar'] = features_polar

    df_shift = asl.df.shift(+1)
    videos = df_shift.index.levels[0]
    for video in videos:
        df_shift.ix[video, 0] = df_shift.ix[video, 1]

    asl.df['delta-rx'] = asl.df['right-x'] - df_shift['right-x']
    asl.df['delta-ry'] = asl.df['right-y'] - df_shift['right-y']
    asl.df['delta-lx'] = asl.df['left-x'] - df_shift['left-x']
    asl.df['delta-ly'] = asl.df['left-y'] - df_shift['left-y']
    features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
    features['delta'] = features_delta

    features_custom = ['grnd-rx', 'grnd-lx', 'delta-ry', 'delta-ly', 'polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
    features['custom'] = features_custom

    return (asl, features)

def train_a_word(word, num_hidden_states, features):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

if __name__ == '__main__':

    #Initializing stuff
    asl = AslDb() # initializes the database

    asl, features = build_features(asl)

    # my_testword = 'CHOCOLATE'
    # print('Training...')
    # model, logL = train_a_word(my_testword, 3, features['ground'])
    # show_model_stats(my_testword, model)
    # print("logL = {}".format(logL))
    # print('Visualizing model data...')
    # visualize(my_testword, model)

    # words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
    # training = asl.build_training(features['ground'])  # Experiment here with different feature sets defined in part 1
    # sequences = training.get_all_sequences()
    # Xlengths = training.get_all_Xlengths()

    # print('Building models with CV selector:')
    # for word in words_to_train:
    #     start = timeit.default_timer()
    #     model = SelectorCV(sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state = 14).select()
    #     end = timeit.default_timer()-start
    #     if model is not None:
    #         print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    #     else:
    #         print("Training failed for {}".format(word))
    # print()
    
    # print('Building models with BIC selector:')
    # for word in words_to_train:
    #     start = timeit.default_timer()
    #     model = SelectorBIC(sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state = 14).select()
    #     end = timeit.default_timer()-start
    #     if model is not None:
    #         print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    #     else:
    #         print("Training failed for {}".format(word))
    # print()

    # print('Building models with DIC selector:')
    # for word in words_to_train:
    #     start = timeit.default_timer()
    #     model = SelectorDIC(sequences, Xlengths, word, 
    #                     min_n_components=2, max_n_components=15, random_state = 14).select()
    #     end = timeit.default_timer()-start
    #     if model is not None:
    #         print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    #     else:
    #         print("Training failed for {}".format(word))
    # print()

    # models = train_all_words(features['ground'], SelectorConstant)
    # print("Number of word models returned = {}".format(len(models)))

    # test_set = asl.build_test(features['ground'])
    # probabilities, guesses = recognize(models, test_set)
    # print("Number of test set items: {}".format(test_set.num_items))
    # print("Number of test set sentences: {}".format(len(test_set.sentences_index)))

    models = train_all_words(features['ground'], SelectorCV)
    test_set = asl.build_test(features['ground'])
    probabilities, guesses = recognize(models, test_set)
    show_errors(guesses, test_set)
