
class Config (object):

    def __init__(self):

        # data parameters
        self.input_variables = 96
        self.seq_length = 1101
        self.num_classes = 2
        self.frame_size = 111
        self.num_frames = 10

        # base encoder configs
        self.num_kernels = 128    
        self.kernel_size = 3
        self.stride = 1
        self.num_conv_layers = 1
        self.conv_dropout = 0.25
        self.conv_padding = 'valid'
        self.embedding_dim = 256
        
        # AR configs
        self.context_dim = 32
        self.num_lstm_layers = 2
        self.lstm_dropout = 0.15

        # Model configs
        self.k_past = 7

        # training configs
        self.num_epoch = 200
        self.batch_size = 128
        self.patience = 10
        self.lambda_1 = 0.3
        self.lambda_2 = 0.7
        self.temperature_coeff = 0.2   #NTXent loss parameter
        self.cosine_similarity = True #NTXent loss parameter

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4
        self.device = 'cuda'
