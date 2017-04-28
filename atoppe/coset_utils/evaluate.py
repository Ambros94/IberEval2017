import argparse
from os.path import isfile

from sklearn.metrics import f1_score


def get_parameters():
    """ Show the arguments help and get the filename of the prediction file."""

    description_string = 'Evaluate the prediction of the COSET dataset.'
    file_description = 'Input file with the prediction for the COSET test.'
    parser = argparse.ArgumentParser(description=description_string)
    parser.add_argument('-p', dest='prediction', required=True,
                        help=file_description)
    parser.add_argument('-g', dest='true_path', required=True,
                        help='File with the groundtrth of the COSET test.')
    args = parser.parse_args()
    return args.prediction, args.true_path


def load_groundtruth(truth_path):
    """ Load the ground truth """
    true_labels, true_ids = [], []
    with open(truth_path) as true_file:
        for line in true_file:
            tweet_id, topic = line.strip().split('\t')
            true_labels.append(int(topic))
            true_ids.append(int(tweet_id))
    return true_ids, true_labels


def load_prediction(filename):
    """ Save the predictions to evaluate the model

    Args:
    filename(str): filename with the predictions

    Return:
    A dictionary with the predictions.
    The keys are the identifiers (int) and the values are the topics (int).
    """
    prediction_dict = {}
    prediction_labels = []
    with open(filename) as pred_file:
        for line in pred_file:
            tweet_id, topic = line.strip().split('\t')
            tweet_id = int(tweet_id)
            prediction_dict[int(tweet_id)] = int(topic)
            prediction_labels.append(int(topic))
    return prediction_dict, prediction_labels


def sort_predictions(true_ids, prediction_dict):
    """ Sort the predictions to follow the same order as the groundtruth

    Args:
    true_ids(list): list of integers with the ids of the groundtruth
    prediction_dict(dict): keys are the tweet ids, values are the topics.

    Returns:
    List with the prediction labels sorted in the same order as in the groundtruth. 
    """
    prediction_labels = []
    for tweet_id in true_ids:
        if tweet_id not in prediction_dict:
            missing_tweet = 'The tweet {} was not found in the prediction file.'
            raise ValueError(missing_tweet.format(tweet_id))
        else:
            prediction_labels.append(prediction_dict[tweet_id])
    return prediction_labels


if __name__ == "__main__":
    # Get the prediction filename
    prediction_path, truth_path = get_parameters()
    if not isfile(prediction_path):
        raise IOError('The prediction file was not found.')
    elif not isfile(truth_path):
        raise IOError('The groundtruth file was not found.')

    # Load the ground truth
    true_ids, true_labels = load_groundtruth(truth_path)

    # Load the prediction filename
    prediction_dict, predi = load_prediction(prediction_path)
    print('Predictions read: {}'.format(len(prediction_dict)))
    prediction = sort_predictions(true_ids, prediction_dict)
    assert len(prediction) == len(prediction_dict)

    if len(prediction) != len(true_labels):
        missing_predictions = '{0} predictions found. {1} labels expected.'
        raise ValueError(missing_predictions.format(len(prediction),
                                                    len(true_labels)))
    # Compute the F1-score
    f1_score = f1_score(true_labels, prediction, average='macro')
    print('* The macro F1-score achieved is: {:.4f}'.format(f1_score))
