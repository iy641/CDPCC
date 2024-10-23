
class Config (object):

    def __init__(self):

        # data parameters
        self.input_variables = 1
        self.seq_length = 5120
        self.num_classes = 3
        self.batch_size = 64
        self.segment_size = 512

        self.num_segments = 10

        # base encoder configs
        self.num_kernels_hidden = 128    #number of 1D kernels in the hidden 1D conv layers
        self.kernel_size = 3
        self.stride = 1
        self.hidden_layers = 1
        self.num_kernels_out = 128   #number of 1D kernels in the output 1D conv layer
        self.conv_dropout = 0.25
        self.conv_padding = 'valid'
        self.features_length = 256

        # Choice of AR model
        self.AR_model = 'LSTM'

        # LSTM configs
        self.context_dim = 32
        self.lstm_layers = 2
        self.lstm_dropout = 0.15

        # Model configs
        self.n_future_timesteps = 2

        # Classifier configs
        self.classifier_input = 'embedding'

        # training configs
        self.num_epoch = 100
        self.patience = 10
        self.lambda_1 = 0.3
        self.lambda_2 = 0.7
        self.temperature_coeff = 0.2   #NTXent loss parameter
        self.cosine_similarity = True #NTXent loss parameter

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4
        self.weight_decay = 3e-4
        self.device = device
