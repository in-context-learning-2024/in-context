


import torch

num_models =5
samples=100
sequence_length=50
size =(num_models,samples,sequence_length)
errs =torch.normal(mean=1, std=1, size=size)
b =10



sample_indices = torch.randint(0, samples, (b, samples)) # Generate all bootstrap samples at once
bootstrap_samples = errs[:,sample_indices,:]
means = bootstrap_samples.mean(dim=2)
variance_estimate = means.var(dim=1)


print("errs:",errs.size())
print("sample_indices:",sample_indices.size())
print("bootstrap_samples:",bootstrap_samples.size())
print("means:",means.size())
print("variance_estimate:",variance_estimate.size())



