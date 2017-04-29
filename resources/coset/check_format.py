import argparse
from os.path import isfile


def get_parameters():
    """ Show the arguments help and get the filename of the prediction file."""

    desc_str = 'Check if the prediction of the file has the correct format.'
    file_description = 'Input file with the prediction for the COSET test.'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-p', dest='prediction', required=True,
                        help=file_description)
    args = parser.parse_args()
    return args.prediction


def check_prediction(filename):
    """ Check if the prediction file has the correct format

    Args:
    filename(str): filename with the predictions

    Return: true if the format is correct or raise an exception

    """
    with open(filename) as pred_file:
        for line in pred_file:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                line_error = 'The format is not correct. There are not two elements each line'
                raise ValueError(line_error)
            else:
                tweet_id, topic = line_split[0], line_split[1]

            if not isinstance(int(tweet_id), int):
                id_int_error = 'The id {} could not be parsed as an int'
                raise ValueError(id_int_error.format(tweet_id))
            elif not isinstance(int(topic), int):
                topic_int_error = 'The topic {} could not be parsed as an int'
                raise ValueError(topic_int_error.format(topic))
    return True


if __name__ == "__main__":
    # Get the prediction filename
    prediction_path = get_parameters()
    if not isfile(prediction_path):
        raise IOError('The prediction file was not found.')

    if check_prediction(prediction_path):
        format_ok = 'The prediction file ({}) has the correct format.'
        print(format_ok.format(prediction_path))
