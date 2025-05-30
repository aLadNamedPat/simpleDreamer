import torch
import torch.nn.functional as F
import torch.nn as nn

class VAE(nn.Module):
    def __init__(
        self, 
        input_channels : int, 
        out_channels : int, 
        latent_dim : int, 
        hidden_dims : list
        ) -> None:

        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        encoder_store = []

        encoder_store.append(
            self.encoder_layer(input_channels, hidden_dims[0])
        )

        for i in range(len(hidden_dims) - 1):
            encoder_store.append(
                self.encoder_layer(hidden_dims[i],
                                    hidden_dims[i + 1])
            )

        self.encoder = nn.Sequential(
            *encoder_store
        )

        self.fc_mu = nn.Linear(self._get_flattened_size(input_channels), latent_dim) #VAE mean calculated
        self.fc_var = nn.Linear(self._get_flattened_size(input_channels), latent_dim) #VAE variance calculated

        decoder_store = []

        #Need to reverse the encoder process to build the decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*144)

        hidden_dims.reverse() #Reverse the hidden state list

        for i in range(len(hidden_dims) - 1):
            decoder_store.append(
                self.decoder_layer(
                    hidden_dims[i],
                    hidden_dims[i + 1]
                )
            )

        self.decoder = nn.Sequential(*decoder_store)

        self.fl = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size = 3,
                stride = 1,
                padding = 1,
                output_padding= 0
            ),
            nn.BatchNorm2d(
                hidden_dims[-1]
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1],
                out_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.Tanh()
        )

    def _get_flattened_size(self, input_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 96, 96)  # Example size, adjust if needed
            dummy_output = self.encoder(dummy_input)
            return torch.flatten(dummy_output, start_dim=1).size(1)
        
    def encoder_layer(
        self,
        input_channels : int,
        output_channels : int,
        ):
        return nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size= 3,
                    stride = 2,
                    padding = 1
                ), 
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU()
        )
    
    def decoder_layer(
        self,
        input_channels : int,
        output_channels : int,
    ):
        #Use ConvTranspose2d to upsample back to the original image size
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding= 1
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )

    def encode(
        self, 
        input : torch.Tensor
    ):
        r = self.encoder(input)
        r = torch.flatten(r, start_dim = 1) #Flatten all dimensions of the encoded data besides the batch size
        u = self.fc_mu(r)
        var = self.fc_var(r)
        return u, var
    
    def reparamterize(
        self,
        u,
        var
    ): #Leveraging the reparametrization trick from the original VAE paper
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)  #reparametrization trick implemented here!
        return eps * std + u #we separate the randomization of z from mu and std which makes it differentiable
    
    def decode(
        self,
        input : torch.Tensor,
        l_hidden_dim : int,
    ):
        a = self.decoder_input(input)
        a = a.view(-1, l_hidden_dim, 12, 12)
        a = self.decoder(a)
        a =  self.fl(a)
        return a

    def forward(
        self,
        input : torch.Tensor,
    ):
        u, var = self.encode(input)
        z = self.reparamterize(u, var)
        a = self.decode(z, 128)
        return a, u, var
    
    def get_latent(
        self,
        input : torch.Tensor,
    ):
        u, var = self.encode(input)
        z = self.reparamterize(u, var)
        return z
    
    #Compute the loss of the encoder-decoder
    def find_loss(
        self,
        reconstructed,
        actual,
        mu,
        log_var,
        kld_weight
    ) -> int:
        recons_loss = F.mse_loss(reconstructed, actual)
        # recons_loss = nn.BCELoss(reconstructed, actual)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu  ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_loss * kld_weight
        return loss