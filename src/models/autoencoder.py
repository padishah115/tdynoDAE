import torch 
import torch.nn as nn
import torch.nn.functional as F

class FcAutoencoder(nn.Module):
    """Fully-Connected autoencoder class responsible for detecting edges in the augmented dataset.


    Attributes
    ----------
        FC1 : nn.Linear
            First layer, which is a fully-connected linear layer: input dimension = 2064*2073, output dimenion = 16
        FC2 : nn.Linear
            Second layer, which is a fully-connected linear layer: input dimension = 16, output dimension 2064*2073
        loss_fn : nn.MSELoss()
            Loss criterion/loss function for the model, here chosen to be the MSE loss.

    
    """

    def __init__(self):
        super().__init__()

        #NAME IT
        self.name = "FcAutoencoder"

        #Introduce two fully-connected layers
        self.FC1 = nn.Linear(in_features=2064*2073, out_features=16)
        self.FC2 = nn.Linear(in_features=16, out_features=2064*2073)

        #Use MSE loss function
        self.loss_fn = nn.MSELoss()

        #Want to flatten input tensors as this autoencoder is fully-connected
        self.flatten = True

    def forward(self, x):
        """Network forward pass for input tensor.
        
        Parameters
        ----------
            x : torch.tensor
                The input layer data of the network

        Returns
        -------
            out : torch.tensor
                Network output of matching dimension compared to input.

        """

        out = F.relu(self.FC1(x))
        out = F.relu(self.FC2(out))

        return out
    

class ConvAutoencoderI(nn.Module):
    """A more sophisticated autoencoder, which uses convolutional layers to produce more informative feature maps and
    representations of the image than fully-connected layers alone."""
    
    def __init__(self, in_depth:int=1, filter_no:int=32):
        """Initialisation function for CONVedgeAutoencoderI class instance

        Parameters
        ----------
            in_depth : int
                Depth (channel number) of the input image. Default is 1 (grayscale image)
            filter_no : int
                Desired number of filters (channels) which we want to produce in the hidden layers after convolution.
        
        """
        super().__init__()

        #NAME IT
        self.name = 'CONVedgeAutoencoderI_0'

        #ENCODER
        self.conv1 = nn.Conv2d(in_channels=in_depth, out_channels=filter_no, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filter_no, out_channels=filter_no, kernel_size=3, padding=1)

        #DECODER
        self.transconv = nn.ConvTranspose2d(in_channels=filter_no, out_channels=filter_no, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(in_channels=filter_no, out_channels=in_depth, kernel_size=3, padding=1)

        #Use the MSE loss function
        self.loss_fn = nn.MSELoss()

        #Don't want to flatten tensors, as we can perform convolution on square data
        self.flatten = False
    
    def forward(self, x):
        """Network forwards pass.

        Parameters:
            x : torch.Tensor()
                Input tensor to the neural network.

        Returns:
            out : torch.Tensor()
                Output tensor after forwards-pass is performed.
        
        """

        #ENCODER
        out = F.max_pool2d(input=F.relu(self.conv1(x)), kernel_size=2)
        out = F.max_pool2d(input=F.relu(self.conv2(out)), kernel_size=2)

        #DECODER
        out = torch.relu(self.transconv(out))
        out = torch.relu(self.transconv(out))
        out = torch.relu(self.final_conv(out))

        return out
