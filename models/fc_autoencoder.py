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

        #ENCODER
        self.FC1 = nn.Linear(in_features=2064*2073, out_features=16)
        
        #DECODER
        self.FC2 = nn.Linear(in_features=16, out_features=2064*2073)

        #Use MSE loss function
        self.loss_fn = nn.MSELoss()

        #Want to flatten input tensors as this autoencoder is fully-connected
        self.flatten = True

    def forward(self, x)->torch.Tensor:
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