import torch
import time
import torch.distributions as dist
import matplotlib.pyplot as plt

class BayesMLR:
    def __init__(self, Y, Z, a_sigma=0.01, b_sigma=0.01, A=1.0, device=None):
        self.b = None
        self.inv_sigma2 = None
        self.inv_tau2 = None
        self.Y = Y.float()
        self.Z = Z.float()
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.A = A
        self.device = device or Y.device

        self.n, self.p = self.Z.shape
        self.q = self.Y.shape[1]
        self._prepare()
        self.posterior_samples = None
        self.sample_time = None
        self.post_mean = None

    def _prepare(self):
        self.U, self.D_vals, Vt = torch.linalg.svd(self.Z, full_matrices=False)
        self.D_vals = self.D_vals.unsqueeze(-1)
        self.D_sq = self.D_vals**2
        self.V = Vt.T
        self.Y_star = self.U.T @ self.Y
        self.D_Y_star = self.D_vals * self.Y_star
        self.Y_sum_sq = torch.sum(self.Y**2)
        self.case_I = self.n >= self.p


        # Initial values
        self.inv_sigma2 = torch.ones(1, self.q, device=self.device)
        self.inv_tau2 = torch.ones(1, self.q, device=self.device)
        self.b = torch.ones(1, self.q, device=self.device)

    def sample(self, num_samples=1000, burnin=100, thinning=1):
        total_iter = burnin + num_samples * thinning
        start = time.time()
        beta_samples = []
        inv_sigma2_samples = []
        inv_tau2_samples = []
        b_samples = []
        mu_samples = []

        for iter in range(total_iter):
            if self.case_I:
                beta_star_mean = self.D_Y_star/ (self.D_sq + self.inv_tau2)
                beta_star_std = torch.sqrt(1.0 /(self.inv_sigma2 * (self.D_sq +  self.inv_tau2)))
                beta_star = beta_star_mean + beta_star_std * torch.randn_like(beta_star_std)
                beta = self.V @ beta_star
                resid = self.Y - self.Z @ beta
                sse = torch.sum(resid**2, dim=0)
                sum_beta_star_sq = torch.sum(beta_star**2, dim=0)
                quad =  sum_beta_star_sq * self.inv_tau2
                self.inv_sigma2 = dist.Gamma(self.a_sigma + 0.5*(self.n + self.p), self.b_sigma + 0.5 * (sse + quad)).sample()
                self.inv_tau2 = dist.Gamma(0.5 + self.p * 0.5, self.b + sum_beta_star_sq * self.inv_sigma2 * 0.5).sample()

            else:
                alpha1 = torch.randn(self.p, 1, device=self.device) / torch.sqrt(self.inv_sigma2 * self.inv_tau2)
                alpha2 = torch.randn(self.n, 1, device=self.device) / torch.sqrt(self.inv_sigma2)
                correction = self.Y_star - self.D_vals * (self.V.T @ alpha1) - alpha2
                correction *=  self.D_vals/ (self.D_sq + self.inv_tau2)
                beta = alpha1 + self.V @ correction
                resid = self.Y - self.Z @ beta
                sum_beta_sq = torch.sum(beta**2, dim=0)
                #print(beta.shape)
                #print(sum_beta_sq.shape)
                quad = torch.sum(resid**2, dim=0)
                quad = quad.view_as(self.inv_tau2)
                #print(quad.shape)
                quad += sum_beta_sq * self.inv_tau2
                self.inv_sigma2 = dist.Gamma(self.a_sigma + (self.n + self.p)*0.5, self.b_sigma + 0.5 * quad).sample()
                self.inv_tau2 = dist.Gamma(0.5 + self.p*0.5, self.b + 0.5 * sum_beta_sq*self.inv_sigma2 ).sample()

            self.b = dist.Gamma(1.0, (1.0 / self.A**2) + self.inv_tau2).sample()

            mu = self.Z @ beta

            if iter >= burnin and (iter - burnin) % thinning == 0:
                mu_samples.append(mu.detach().clone())
                beta_samples.append(beta.detach().clone())
                inv_sigma2_samples.append(self.inv_sigma2.detach().clone())
                inv_tau2_samples.append(self.inv_tau2.detach().clone())
                b_samples.append(self.b.detach().clone())

        end = time.time()

        self.posterior_samples = {
            "mu": torch.stack(mu_samples),
            "beta": torch.stack(beta_samples),
            "inv_sigma2": torch.stack(inv_sigma2_samples),
            "inv_tau2": torch.stack(inv_tau2_samples),
            "b": torch.stack(b_samples)
        }

        self.sample_time = end - start

    def posterior_mean(self):
        self.post_mean = {}
        for para, sample in self.posterior_samples.items() :
            if para == "beta" or para == "mu":
                self.post_mean[para] = torch.mean(sample, dim=0).squeeze()
            else:
                self.post_mean[para] = torch.mean(sample)

        return self.post_mean


    def posterior_predictive(self, Z_new):
        """
        Generate posterior predictive draws for new covariate matrix Z_new.

        Returns: Tensor of shape (num_samples, n_new)
        """
        if self.posterior_samples is None:
            raise RuntimeError("Run `sample()` before posterior predictive.")

        beta_samples = self.posterior_samples["beta"]  # (S, p, 1)
        inv_sigma2_samples = self.posterior_samples["inv_sigma2"]  # (S,)

        S = beta_samples.shape[0]
        Z_new = Z_new.float().to(self.device)  # (n_new, p)

        Y_pred = []
        for s in range(S):
            mu = Z_new @ beta_samples[s].squeeze()  # (n_new,)
            eps = torch.randn_like(mu) / torch.sqrt(inv_sigma2_samples[s])
            Y_pred.append(mu + eps)

        return torch.stack(Y_pred)  # (S, n_new)

    def plot_trace(self, param: str, idx: int = None):
        """
        Plot trace plot of a parameter.
        - param: "sigma2", "tau2", "b", or "beta"
        - idx: index for beta[k], only needed if param == "beta"
        """
        import matplotlib.pyplot as plt

        if self.posterior_samples is None:
            raise RuntimeError("Run `sample()` first.")

        samples = self.posterior_samples[param]
        if param == "beta":
            if idx is None:
                raise ValueError("Must specify idx for beta[k]")
            series = samples[:, idx].squeeze().cpu()
        else:
            series = samples.cpu()

        plt.plot(series)
        plt.title(f"Trace plot for {param}{'' if idx is None else f'[{idx}]'}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

    def plot_posterior(self, param: str, idx: int = None, bins=50):
        """
        Plot posterior histogram/density.
        """
        import matplotlib.pyplot as plt

        if self.posterior_samples is None:
            raise RuntimeError("Run `sample()` first.")

        samples = self.posterior_samples[param]
        if param == "beta":
            if idx is None:
                raise ValueError("Must specify idx for beta[k]")
            series = samples[:, idx].squeeze().cpu()
        else:
            series = samples.cpu()

        plt.hist(series.numpy(), bins=bins, density=True, alpha=0.7)
        plt.title(f"Posterior distribution of {param}{'' if idx is None else f'[{idx}]'}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()