if (self.input_noise == True):
    state += torch.randn_like(state) * self.noise
    print("Input noise added")


#DONE: Add if statement for weight_one_noise so this block happens when weight_one_noise is true
'''
if (self.weight_one_noise == True):
    for param in self.policy[0].modules():
        #if isinstance(param, self.policy.nn.Linear):
        param.bias.data += torch.randn_like(param.bias.data) * self.noise
        param.weight.data += torch.randn_like(param.weight.data) * self.noise
        #print("Weight one noise added")
'''

if (self.weight_one_noise == True):
    for param in self.policy[0].modules():
        #if isinstance(param, self.policy.nn.Linear):
        mean = 0
        # Generate the Gaussian noise tensor
        noiseTensor = torch.tensor(np.random.normal(mean, self.noise, param.weight.data.size()), device=self.device, dtype=torch.float)

        # Add the noise tensor to the original tensor
        param.weight.data = param.weight.data + noiseTensor

if (self.weight_two_noise == True):
    for param in self.policy[2].modules():
        #if isinstance(param, self.policy.nn.Linear):
        mean = 0
        # Generate the Gaussian noise tensor
        noiseTensor = torch.tensor(np.random.normal(mean, self.noise, param.weight.data.size()), device=self.device, dtype=torch.float)

        # Add the noise tensor to the original tensor
        param.weight.data = param.weight.data + noiseTensor

'''
#DONE: Add if statement for weight_two_noise so this block happens when is true
if (self.weight_two_noise == True):
    for param in self.policy[2].modules():
        #if isinstance(param, self.policy.nn.Linear):
        param.bias.data += torch.randn_like(param.bias.data) * self.noise
        param.weight.data += torch.randn_like(param.weight.data) * self.noise
        #print("Weight two noise added")
'''
# Parameterize distribution with policy, sample action
normal_dist = self.distribution(self.policy(state), self.logstd.exp())
action = normal_dist.sample()