import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoderI(nn.Module):
    """A more sophisticated autoencoder, which uses convolutional layers to produce more informative feature maps and
    representations of the image than fully-connected layers alone."""
    
    def __init__(self, in_depth:int=1, filter_no:int=32):
        """Initialisation function for CONVedgeAutoencoderI class instance.

        Parameters
        ----------
            in_depth : int
                Depth (channel number) of the input image. Default is 1 (grayscale image)
            filter_no : int
                Desired number of filters (channels) which we want to produce in the hidden layers after convolution.
        
        """
        
        # CALL INIT FUNCTION FOR NN.MODULE CLASS
        super().__init__()

        #NAME IT!
        self.name = 'CONVedgeAutoencoderI_0'

        #ENCODER LAYERS
        self.conv1 = nn.Conv2d(in_channels=in_depth, out_channels=filter_no, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filter_no, out_channels=filter_no, kernel_size=3, padding=1)

        #DECODER LAYERS
        self.transconv = nn.ConvTranspose2d(in_channels=filter_no, out_channels=filter_no, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(in_channels=filter_no, out_channels=in_depth, kernel_size=3, padding=1)

        #Use the MSE loss function
        self.loss_fn = nn.MSELoss()

        #Don't want to flatten tensors, as we can perform convolution on square data
        self.flatten = False
    
    def forward(self, x)->torch.Tensor:
        """Network forwards pass.

        Parameters:
            x : torch.Tensor()
                Input tensor to the neural network. Should be a grayscale image, with channel number equal to 1

        Returns:
            out : torch.Tensor()
                Output tensor after forwards-pass is performed.
        
        """

        #ENCODER LAYERS
        out = F.max_pool2d(input=F.relu(self.conv1(x)), kernel_size=2)
        out = F.max_pool2d(input=F.relu(self.conv2(out)), kernel_size=2)

        #DECODER LAYERS
        out = torch.relu(self.transconv(out))
        out = torch.relu(self.transconv(out))
        out = torch.relu(self.final_conv(out))

        return out
