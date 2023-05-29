import torch
import torch.nn.functional as F
import torch.distributions as dist
import torch.nn as nn
import copy



def featurize(model, x, probabilistic=True, z_dim=31):
    if not probabilistic:
        return model(x)
    else:
        z_params = model(x)
        z_mu = z_params[:, :z_dim] #anche qua è il punto cruciale
        z_sigma = F.softplus(z_params[:, z_dim:])
        z_dist = dist.Independent(dist.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([1]).view([-1, z_dim])
        
        return z,(z_mu,z_sigma)



def compute_loss(model,criterion, optim, dataset,device,lr=0.001, num_classes=62, z_dim=32,L2R_coeff=0.01,CMI_coeff=0.001):
    cls= nn.Linear(z_dim,num_classes)
    cls.to(device)
    model = copy.deepcopy(nn.Sequential(model,cls))
    model.to(device)
    model.train()

    total_loss = 0
    total_regL2R = 0
    total_regCMI = 0
    total_regNegEnt = 0
    total_samples = 0
    correct=0
    
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    r_mu = nn.Parameter(torch.zeros(num_classes,z_dim))
    r_sigma = nn.Parameter(torch.ones(num_classes,z_dim))
    C = nn.Parameter(torch.ones([]))
    optim.add_param_group({'params':[r_mu,r_sigma,C],'lr':lr,'momentum':0.9})

    for x, y in dataset:
        x, y = x.to(device), y.to(device)

        z, (z_mu, z_sigma) = model.featurize(x, return_dist=True)
        logits = cls(z)
        
        loss = criterion(logits, y) #qua è il punto cruciale

        obj = loss
        regL2R = torch.zeros_like(obj)
        regCMI = torch.zeros_like(obj)
        regNegEnt = torch.zeros_like(obj)
        

        if L2R_coeff != 0.0:
            regL2R = z.norm(dim=1).mean()
            obj = obj + L2R_coeff * regL2R

        if CMI_coeff != 0.0:
            r_sigma_softplus = F.softplus(r_sigma)
            r_mu = r_mu[y]
            r_sigma = r_sigma_softplus[y]
            z_mu_scaled = z_mu * C
            z_sigma_scaled = z_sigma * C
            regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                     (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
            regCMI = regCMI.sum(1).mean()
            obj = obj + CMI_coeff * regCMI

        z_dist = dist.Independent(dist.Normal(z_mu, z_sigma), 1)
        mix_coeff = dist.Categorical(x.new_ones(x.shape[0]))
        mixture = dist.MixtureSameFamily(mix_coeff, z_dist)
        log_prob = mixture.log_prob(z)
        regNegEnt = log_prob.mean()

        optim.zero_grad()
        obj.backward()
        optim.step()


        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_regL2R += regL2R.item() * batch_size
        total_regCMI += regCMI.item() * batch_size
        total_regNegEnt += regNegEnt.item() * batch_size
        total_samples += y.size(0)
        _, prediction = torch.max(logits.data, 1)
        correct += (prediction == y).sum().item()

    loss_avg = total_loss / total_samples
    regL2R_avg = total_regL2R / total_samples
    regCMI_avg = total_regCMI / total_samples
    regNegEnt_avg = total_regNegEnt / total_samples
    acc=correct/total_samples

    return regL2R_avg, regCMI_avg, regNegEnt_avg ,acc, loss_avg, model
