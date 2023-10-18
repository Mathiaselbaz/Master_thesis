class ClassCondFlow(nn.Module):
    """
    Class conditional normalizing Flow model
    """

    def __init__(self, q0, flows):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.q_base=distributions.DiagGaussian(3, trainable=False)
        self.add_flow=True

    def forward_kld(self, x, y):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        num_samples=x.shape[1]
        z = x.view(-1, *x.size()[2:])
        log_q = torch.zeros(len(z), dtype=x.dtype, device=x.device)
        z = z.view(-1, num_samples, *z.size()[1:])
        log_q = log_q.view(-1, num_samples, *log_q.size()[1:])
        z,log_det = self.q0.log_prob(z,y)
        print('z before NSF :',torch.mean(z[0,:,:].clone(),0))
        log_q += log_det
        z = z.view(-1, *x.size()[2:])
        if self.add_flow : 
            log_q_bis= torch.zeros(len(z), dtype=x.dtype, device=x.device)
            for i in range(len(self.flows) - 1, -1, -1):
                z, log_det = self.flows[i].inverse(z)
                log_q_bis += log_det
            log_q += log_q_bis.view(-1, num_samples, *log_q_bis.size()[1:])
        log_p0 = self.q_base.log_prob(z)
        log_q += log_p0.view(-1, num_samples, *log_p0.size()[1:])

        return -torch.mean(log_q)

    def sample(self, num_samples=1, y=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None

        Returns:
          Samples, log probability
        """
        z, log_q = self.q_base(num_samples)
        if self.add_flow:
            for flow in self.flows:
                z, log_det = flow(z)
                log_q -= log_det
        z, log_det = self.q0.forward2(y, z.unsqueeze(0).unsqueeze(2))
        print('Size of log_det :', log_det.size())
        log_q -= log_det
        return z, log_q

    def log_prob(self, x, y):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        z, log_q = self.q0(x, num_samples=num_samples)
        # Flatten batch and sample dim
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det           
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
         param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))
        
    def load2(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

