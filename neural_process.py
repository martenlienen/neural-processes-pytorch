import torch
import torch.nn as nn


def kullback_leiber_divergence(mu, sigma, nu, tau):
    """Compute the Kullback-Leibler divergence between two univariate normals

    D(Q || P) where Q ~ N(mu, sigma^2) and P ~ N(nu, tau^2)

    See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    """
    sigma = sigma + 1e-16
    tau = tau + 1e-16

    return ((sigma**2 + (mu - nu)**2) / tau**2 - 1 + 2 * torch.log(tau / sigma)).sum() / 2


def log_likelihood(mu, sigma, x):
    """Compute the log-likelihood of n x under m univariate normal distributions

    This omits a constant log(2pi) term.

    Parameters
    ----------
    mu : torch.Tensor of shape m
    sigma : torch.Tensor of shape m
    x : torch.Tensor of shape n

    Returns
    -------
    torch.Tensor of shape [n, m]
    """
    # Unsqueeze everything to ensure correct broadcasting
    x = x.unsqueeze(dim=1)
    mu = mu.unsqueeze(dim=0)
    sigma = sigma.unsqueeze(dim=0)

    return -(x - mu)**2 / (2 * sigma**2) - torch.log(sigma)


class NeuralProcess(nn.Module):
    """A neural process as defined by Garnelo et al. [1]

    [1] https://arxiv.org/abs/1807.01622
    """

    def __init__(self, encoder, aggregator, z_decoder, decoder):
        """Create a new neural process model.

        Parameters
        ----------
        encoder : torch.nn.Module
            Module that encodes batches of x and y into r vectors
        aggregator : torch.nn.Module
            Module that combines a batch of r vectors into a single one
        z_decoder : torch.nn.Module
            Module that decodes an r vector into parameters mu and sigma of a Gaussian RV
        decoder : torch.nn.Module
            Module that decodes a batch of x and a batch of z into batches of parameters for a Gaussian RV
        """
        super().__init__()

        self.encoder = encoder
        self.aggregator = aggregator
        self.z_decoder = z_decoder
        self.decoder = decoder

        self.std_normal = torch.distributions.Normal(0.0, 1.0)
        self.elbo_samples = 10

    def forward(self, x_context, y_context, x_target):
        # Encode the context
        r_context = self.aggregator(self.encoder(x_context, y_context))

        # Compute the parameters of the latent variable
        mu, sigma = self.z_decoder(r_context)

        # Sample z with the reparameterisation trick so that we can backprop through mu and sigma\
        z = self.std_normal.sample() * sigma + mu
        z = z.unsqueeze(0)

        # Run the conditional decoder for all targets
        mu, sigma = self.decoder(x_target, z)

        return mu, sigma

    def elbo(self, x, y, x_context, y_context, x_target, y_target):
        """Compute the evidence lower-bound for training."""
        # Find q(z | x, y)
        mu, sigma = self.z_decoder(self.aggregator(self.encoder(x, y)))

        # Find q(z | x_context, y_context)
        mu_ctx, sigma_ctx = self.z_decoder(self.aggregator(self.encoder(x_context, y_context)))

        # Compute the Kullback-Leibler divergence part
        kld = kullback_leiber_divergence(mu, sigma, mu_ctx, sigma_ctx)

        # Estimate the log-likelihood part of the ELBO by sampling z from q(z | x, y)
        z = torch.distributions.Normal(mu, sigma).sample((self.elbo_samples,))
        mu, sigma = self.decoder(x_target, z)
        log_llh = log_likelihood(mu, sigma, y_target).sum(dim=0).mean()

        return log_llh - kld
