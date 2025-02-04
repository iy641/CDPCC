
class Config (object):

    def __init__(self):

        # data parameters
        self.input_variables = 1
        self.seq_length = 5120
        self.num_classes = 3
        self.frame_size = 512
        self.num_frames = 10
        
        # base encoder configs
        self.num_kernels = 128    
        self.kernel_size = 3
        self.stride = 1
        self.num_conv_layers = 1
        self.conv_dropout = 0.25
        self.conv_padding = 'valid'
        self.embedding_dim = 256


        # LSTM configs
        self.context_dim = 32
        self.num_lstm_layers = 2
        self.lstm_dropout = 0.15

        # Model configs
        self.k_past = 8

        # training configs
        self.num_epoch = 100
        self.batch_size = 64
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
        self.device = 'cuda'
