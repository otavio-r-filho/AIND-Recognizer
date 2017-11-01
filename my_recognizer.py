import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    Xlengths = test_set.get_all_Xlengths()
    score = None
    
    for item in range(test_set.num_items):
        X_seq, seq_len = Xlengths[item]
        word_scores = dict()
        word_gess = (float('-inf'), None)
        for word, model in models.items():
            try:
                score = models[word].score(X_seq, seq_len)
            except:
                score = None

            if score is not None:
                word_scores[word] = score
                if score > word_gess[0]:
                    word_gess = (score, word)

        probabilities.append(word_scores)
        guesses.append(word_gess[1])

    return (probabilities, guesses)
