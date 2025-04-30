import argparse
def get_config():
    # Parameter management
    parse = argparse.ArgumentParser(description='peptide default main')  # Creates an ArgumentParser object parse as a command line parser, and sets the description of the parser.
    parse.add_argument('-task', type=str, default='model_select')  # Adds a parameter named -task and sets its default value to 'model_select'.
    # Model training parameters
    parse.add_argument('-model', type=str, default='TextCNN',
                       help='The name of model')
    parse.add_argument('--criterion', nargs='+', type=str, default=['FDL','DBL'],
                       help='{CL: Combo loss, FDL: Focal dice loss, DL: Dice loss, CE: Cross entropy loss, '
                            'ASL: Asymmetric loss, FL: Focal loss} ')
    parse.add_argument('--loss_weights', nargs='+', type=float, default=[0.99,0.01],
                       help='Weights for each loss function. Must match the number of criterion. '
                            'e.g., --loss_weights 0.7 0.3 0.3 0.2')
    parse.add_argument('-subtest', type=bool, default=False)
    parse.add_argument('-vocab_size', type=int, default=21,
                       help='The size of the vocabulary')
    parse.add_argument('-output_size', type=int, default=21,
                       help='Number of peptide functions')  # Number of peptides
    parse.add_argument('-batch_size', type=int, default=64*4,
                       help='Batch size')
    parse.add_argument('-epochs', type=int, default=200)
    parse.add_argument('-learning_rate', type=float, default=0.0007)
    # Deep model parameters
    parse.add_argument('-embedding_size', type=int, default=64*2,
                       help='Dimension of the embedding')
    parse.add_argument('-dropout', type=float, default=0.6)
    parse.add_argument('-filter_num', type=int, default=64*2,
                       help='Number of the filter')  # Number of filters
    parse.add_argument('-filter_size', type=list, default=[3,4,5],
                       help='Size of the filter')
    # File paths
    parse.add_argument('-model_path', type=str, default='saved_models/model_select+TextCNN0.pth',
                       help='Path of the training data')
    parse.add_argument('-train_direction', type=str, default='dataset/augmented_train.txt',
                       help='Path of the training data')
    parse.add_argument('-test_direction', type=str, default='dataset/test.txt',
                       help='Path of the test data')
    parse.add_argument('-threshold_percentile', type=int, default=40)
    config = parse.parse_args()  # Parse the added arguments
    return config

if __name__ == '__main__':
    args = get_config()                  # Get parameters from the configuration file
    from train_test import TrainAndTest
    TrainAndTest(args)