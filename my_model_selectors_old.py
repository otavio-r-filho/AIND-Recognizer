import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """

        # model.n_components
        # model.n_features
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            # bic, hmm_model = GaussianHMM(n_components=self.min_n_components, covariance_type="diag", n_iter=1000,
            #                              random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            bic, hmm_model = self.build_model(self.min_n_components, self.X, self.lengths)

            for n_comp in range(self.min_n_components + 1, self.max_n_components + 1):
                candidate_bic, candidate_model = self.build_model(n_comp, self.X, self.lengths)
                if candidate_bic > bic:
                    break
                bic = candidate_bic
                hmm_model = candidate_model

            return hmm_model

        except:
            #The exception will return a model if the number of components have been tested
            if n_comp == self.max_n_components:
                if hmm_model:
                    return hmm_model
                else:
                    return None

    def build_model(self, n_comp, X_seq, seq_len):
        hmm_model = GaussianHMM(n_components=self.min_n_components, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        logL = hmm_model.score(X_seq, seq_len)
        p = hmm_model.n_features
        N = hmm_model.n_components
        bic = (-2 * logL) + (p * np.log(N))
        return bic, hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        log_sum = 0.0
        models = []
        dics = []
        alpha = 1.0 / ( self.max_n_components - self.min_n_components - 1.0)

        try:
            for n_comp in range(self.min_n_components, self.max_n_components + 1):
                logL, model = self.build_model(n_comp, self.X, self.lengths)
                if model is not None:
                    models.append((logL, model))
                    log_sum += logL
            
            if len(models):
                for logL, model in models:
                    dic = logL - (alpha * (log_sum - logL))
                    dics.append((dic, model))

                # The model with the largest DIC will be chosen
                dic, model = max(dics)
                return model
            return None
        except:
            return None

        # TODO implement model selection based on DIC scores
        raise NotImplementedError

    def build_model(self, n_comp, X_seq, seq_len):
        hmm_model = GaussianHMM(n_components=self.min_n_components, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        logL = hmm_model.score(X_seq, seq_len)
        return logL, hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        num_seqs = 1

        for n in range(5, 1, -1):
            if (len(self.sequences) / n) >= 2.0:
                num_seqs = n
                break
            if n == 2 and (len(self.sequences) / n) >= 1.0:
                num_seqs = n

        cv = KFold(n_splits=num_seqs, shuffle=True, random_state=self.random_state)

        models = []

        for cv_train, cv_test in cv.split(self.sequences):
            X_train, train_length = combine_sequences(cv_train, self.sequences)
            X_test, test_length = combine_sequences(cv_test, self.sequences)
            model, logL = self.build_model(X_train, train_length, X_test, test_length)
            if model and logL:
                models.append((logL, model))

        Lhood, hmm_model = max(models)
        return hmm_model

    def build_model(self, X_train_seq, train_seq_len, X_test_seq, test_seq_len):
        '''
        This method will select the best model within the n_components range
        '''

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=self.min_n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train_seq, train_seq_len)
            logL = hmm_model.score(X_test_seq, test_seq_len)

            for n_comp in range(self.min_n_components + 1, self.max_n_components + 1):
                candidate_model = GaussianHMM(n_components=n_comp, covariance_type="diag", n_iter=1000,
                                              random_state=self.random_state, verbose=False).fit(X_train_seq, train_seq_len)
                candidate_model_logL = candidate_model.score(X_test_seq, test_seq_len)

                if candidate_model_logL < logL:
                    break
                hmm_model = candidate_model
                logL = candidate_model_logL

            return hmm_model, logL

        except:
            if hmm_model and logL:
                return hmm_model, logL
            return None
