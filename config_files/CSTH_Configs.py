
class Config (object):

    def __init__(self):

        # data parameters
        self.input_variables = 3
        self.frame_size = 20
        self.seq_length = 200    # size of raw time series
        self.batch_size = 128

        self.num_frames = 10

        # base encoder configs
        self.num_kernels_hidden = 128    #number of 1D kernels in the hidden 1D conv layers
        self.kernel_size = 3
        self.stride = 1
        self.hidden_layers = 2
        self.num_kernels_out = 128   #number of 1D kernels in the output 1D conv layer
        self.conv_dropout = 0.25
        self.conv_padding = 'valid'
        self.features_length = 256

        # Choice of AR model
        self.AR_model = 'LSTM'

        # LSTM configs
        self.context_dim = 64
        self.lstm_layers = 1
        self.lstm_dropout = 0.15

        # Transformer configs
        self.context_dim = 64
        self.depth = 4
        self.n_heads = 4
        self.mlp_dim = 64
        self.transformer_dropout = 0.25

        # Model configs
        self.n_future_timesteps = 3

        # Classifier configs
        self.classifier_input = 'embedding'
        self.num_classes = 2

        # training configs
        self.num_epoch = 200
        self.patience = 10
        self.lambda_1 = 0.3
        self.lambda_2 = 0.7
        self.temperature_coeff = 0.2   #NTXent loss parameter
        self.cosine_similarity = True #NTXent loss parameter

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr =0.001
        self.device = 'cuda'
