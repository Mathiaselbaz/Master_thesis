import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize 


class BaseEncoder(nn.Module):
    """
    Base distribution of a flow-based variational autoencoder
    Parameters of the distribution depend of the target variable x
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        """
        Args:
          x: Variable to condition on, first dimension is batch size
          num_samples: number of samples to draw per element of mini-batch

        Returns
          sample of z for x, log probability for sample
        """
        raise NotImplementedError

    def log_prob(self, z, x):
        """

        Args:
          z: Primary random variable, first dimension is batch size
          x: Variable to condition on, first dimension is batch size

        Returns:
          log probability of z given x
        """
        raise NotImplementedError


class Dirac(BaseEncoder):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1)
        log_p = torch.zeros(z.size()[0:2])
        return z, log_p

    def log_prob(self, z, x):
        log_p = torch.zeros(z.size()[0:2])
        return log_p


class Uniform(BaseEncoder):
    def __init__(self, zmin=0.0, zmax=1.0):
        super().__init__()
        self.zmin = zmin
        self.zmax = zmax
        self.log_p = -torch.log(zmax - zmin)

    def forward(self, x, num_samples=1):
        z = (
            x.unsqueeze(1)
            .repeat(1, num_samples, 1)
            .uniform_(min=self.zmin, max=self.zmax)
        )
        log_p = torch.zeros(z.size()[0:2]).fill_(self.log_p)
        return z, log_p

    def log_prob(self, z, x):
        log_p = torch.zeros(z.size()[0:2]).fill_(self.log_p)
        return log_p


class ConstDiagGaussian(BaseEncoder):
    def __init__(self, loc, scale):
        """Multivariate Gaussian distribution with diagonal covariance and parameters being constant wrt x

        Args:
          loc: mean vector of the distribution
          scale: vector of the standard deviations on the diagonal of the covariance matrix
        """
        super().__init__()
        self.d = len(loc)
        if not torch.is_tensor(loc):
            loc = torch.tensor(loc).float()
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale).float()
        self.loc = nn.Parameter(loc.reshape((1, 1, self.d)))
        self.scale = nn.Parameter(scale)

    def forward(self, x=None, num_samples=1):
        """
        Args:
          x: Variable to condition on, will only be used to determine the batch size
          num_samples: number of samples to draw per element of mini-batch

        Returns:
          sample of z for x, log probability for sample
        """
        if x is not None:
            batch_size = len(x)
        else:
            batch_size = 1
        eps = torch.randn((batch_size, num_samples, self.d), device=x.device)
        z = self.loc + self.scale * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            torch.log(self.scale) + 0.5 * torch.pow(eps, 2), 2
        )
        return z, log_p

    def log_prob(self, z, x):
        """
        Args:
          z: Primary random variable, first dimension is batch dimension
          x: Variable to condition on, first dimension is batch dimension

        Returns:
          log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            torch.log(self.scale) + 0.5 * ((z - self.loc) / self.scale) ** 2, 2
        )
        return log_p


class NNDiagGaussian(BaseEncoder):
    """
    Diagonal Gaussian distribution with mean and variance determined by a neural network
    """

    def __init__(self, net):
        """Construtor

        Args:
          net: net computing mean (first n / 2 outputs), standard deviation (second n / 2 outputs)
        """
        super().__init__()
        self.net = net

    def forward(self, x, num_samples=1):
        """
        Args:
          x: Variable to condition on
          num_samples: number of samples to draw per element of mini-batch

        Returns:
          sample of z for x, log probability for sample
        """
        batch_size = len(x)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        std = torch.exp(0.5 * mean_std[:, n_hidden : (2 * n_hidden), ...].unsqueeze(1))
        
        eps = torch.randn(
            (batch_size, num_samples) + tuple(mean.size()[2:]), device=x.device
        )
        z = (mean + std * eps)
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(
            2 * np.pi
        ) - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(2, z.dim())))
        return z.squeeze(0), log_p.squeeze(0)

    def log_prob(self, z, x):
        """

        Args:
          z: Primary random variable, first dimension is batch dimension
          x: Variable to condition on, first dimension is batch dimension

        Returns:
          log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        var = torch.exp(mean_std[:, n_hidden : (2 * n_hidden), ...].unsqueeze(1))
        print('mean :',torch.mean(mean.squeeze(0),0))
        print('var :',torch.mean(var.squeeze(0),0))
        print('first term :',torch.mean(0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi)))
        print('second term :',torch.mean(0.5 * torch.sum(torch.log(var) + (z - mean) ** 2 / var, 2)))
        print('shape :',(torch.log(var) + (z - mean) ** 2).size())
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(
            2 * np.pi
        ) - 0.5 * torch.sum(torch.log(var) + (z - mean) ** 2 / var, 2)
        return log_p

    
def chi_square(beta, hist,n_beta,nom_prob): 
    prob_grid=torch.sum(torch.Tensor(beta).unsqueeze(0).unsqueeze(0).expand(200, 200, n_beta).to(nom_prob.device)*nom_prob,dim=-1)*50000
    # Calculate the residuals 
    chi2 = torch.sum((hist.to(nom_prob.device) - prob_grid)**2) 
    return chi2 


def chi_2(hist,n_beta,nom_prob):
    initial_beta = [1,1,1] 
    Beta=torch.zeros(hist.shape[0], n_beta) 
    X=hist.clone().to(torch.device('cpu'))
    def objective(beta): 
        return chi_square(beta, X, n_beta,nom_prob) 
    # Perform the optimization 
    result = minimize(objective, initial_beta, method='Nelder-Mead') 
    Beta=torch.Tensor(result.x) 
    return Beta
    
class Chi2DiagGaussian(BaseEncoder):
    """
    Diagonal Gaussian distribution with mean and variance determined by a neural network
    """

    def __init__(self, n_beta):
        """Construtor

        Args:
          net: net computing mean (first n / 2 outputs), standard deviation (second n / 2 outputs)
        """
        super().__init__()
        self.n_beta=n_beta

    def forward(self, x, num_samples=1):
        """
        Args:
          x: Variable to condition on
          num_samples: number of samples to draw per element of mini-batch

        Returns:
          sample of z for x, log probability for sample
        """
        batch_size=1
        beta0=torch.mean(x[0,:,:].clone().unsqueeze(0),1).to(torch.device('cpu'))
        mean_std = torch.cat((beta0,torch.Tensor([-5,-5,-5],device=torch.device('cpu')).unsqueeze(0).repeat(beta0.shape[0],1)),1).to(x.device)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        std = torch.exp(0.5 * mean_std[:, n_hidden : (2 * n_hidden), ...].unsqueeze(1))
        eps = torch.randn(
            (batch_size, num_samples) + tuple(mean.size()[2:]), device=x.device
        )
        z = (mean + std * eps)
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(
            2 * np.pi
        ) - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(2, z.dim())))
        return z.squeeze(0), log_p.squeeze(0)

    def log_prob(self, z, x):
        """

        Args:
          z: Primary random variable, first dimension is batch dimension
          x: Variable to condition on, first dimension is batch dimension

        Returns:
          log probability of z given x
        """
        beta0=torch.mean(x,1).to(torch.device('cpu'))
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        mean_std = torch.cat((beta0,torch.Tensor([-5,-5,-5],device=torch.device('cpu')).unsqueeze(0).repeat(beta0.shape[0],1)),1).to(x.device)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        mean=beta0.unsqueeze(1).repeat(1,z.shape[1],1).to(x.device)
        var = torch.exp(mean_std[:, n_hidden : (2 * n_hidden), ...].unsqueeze(1))
        print('mean :',torch.mean(mean.squeeze(0),0))
        print('var :',torch.mean(var.squeeze(0),0))
        print('first term :',torch.mean(0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi)))
        print('second term :',torch.mean(0.5 * torch.sum(torch.log(var) + (z - mean) ** 2 / var, 2)))
        print('shape :',(torch.log(var) + (z - mean) ** 2).size())
        print('z :',torch.mean(z.squeeze(0),0))
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(
            2 * np.pi
        ) - 0.5 * torch.sum( (z - mean) ** 2 / var, -1)
        return log_p

def Cholesky(cholesky_tensor,n_beta):
    L=torch.zeros(cholesky_tensor.shape[0],n_beta,n_beta)
    for i in range(n_beta):
        for j in range(n_beta):
            if i==j:
                L[:,i,j]=torch.abs(cholesky_tensor[:,i])
            if j<i:
                L[:,i,j]=-cholesky_tensor[:,n_beta-1+i+j]
    return L.to(cholesky_tensor.device)
    
class MyGaussianFlow(BaseEncoder):
    """
    Multivariate Gaussian distribution with mean and variance determined by a neural network
    """

    def __init__(self, net):
        """Construtor

        Args:
          net: net computing mean (first n / 2 outputs), standard deviation (second n / 2 outputs)
        """
        super().__init__()
        self.net = net
        self.n_beta=3

    def forward(self, x, num_samples=1):
        """
        Args:
          x: Variable to condition on
          num_samples: number of samples to draw per element of mini-batch

        Returns:
          sample of z for x, log probability for sample
        """
        batch_size = len(x)
        mean_std = self.net(x)
        mean = mean_std[:, :self.n_beta].unsqueeze(1).unsqueeze(1)
        mean_std[:, self.n_beta:2*self.n_beta]=mean_std[:, self.n_beta:2*self.n_beta]
        cholesky=Cholesky(torch.exp(mean_std[:, self.n_beta:]),self.n_beta).unsqueeze(1)
        eps = torch.randn((batch_size, num_samples,1,self.n_beta), device=x.device)
        z = (mean + torch.matmul(eps,cholesky)).squeeze(2)
        log_p = - torch.sum((mean_std[:, self.n_beta:2*self.n_beta]).unsqueeze(2),1) #- 0.5*torch.sum(eps**2,-1)
        return z.squeeze(2).squeeze(0), log_p.squeeze(0).squeeze(0).squeeze(1)
    
    def forward2(self, x, z):
        """
        Args:
          x: Variable to condition on
          num_samples: number of samples to draw per element of mini-batch

        Returns:
          sample of z for x, log probability for sample
        """
        batch_size = len(x)
        mean_std = self.net(x)
        
        mean = mean_std[:, :self.n_beta].unsqueeze(1).unsqueeze(1)
        mean_std[:, self.n_beta:2*self.n_beta]=mean_std[:, self.n_beta:2*self.n_beta]
        cholesky=Cholesky(torch.exp(mean_std[:, self.n_beta:]),self.n_beta).unsqueeze(1)
        z = (mean + torch.matmul(z,cholesky)).squeeze(2).to(x.device)
        del(x,cholesky)
        log_p =  - torch.sum((mean_std[:, self.n_beta:2*self.n_beta]).unsqueeze(2),1) 
        return z.squeeze(0), log_p.squeeze(0)


    def log_prob(self, z, x):
        """

        Args:
          z: Primary random variable, first dimension is batch dimension
          x: Variable to condition on, first dimension is batch dimension

        Returns:
          log probability of z given x
        """
        batch_size = len(x)
        mean_std = self.net(x)
        del(x)
        mean = mean_std[:, :self.n_beta].unsqueeze(1)
        cholesky=Cholesky(torch.exp(mean_std[:, self.n_beta:]),self.n_beta).unsqueeze(1)
        Cov=torch.matmul(cholesky,torch.transpose(cholesky,2,3))
        eps=torch.matmul((z-mean).unsqueeze(2),torch.linalg.inv(cholesky)).squeeze(2)
        del(cholesky,z,Cov,mean)
        print('first term :',torch.mean(torch.sum((mean_std[:, self.n_beta:2*self.n_beta]).unsqueeze(2),1)))
        print('second term :',torch.mean(0.5*torch.sum(eps**2,-1)))
        log_p = -0.5 * self.n_beta * np.log(2 * np.pi) - torch.sum((mean_std[:, self.n_beta:2*self.n_beta]).unsqueeze(2),1)
        return eps,log_p