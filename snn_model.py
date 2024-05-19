import torch
import snntorch as snn
import torch.nn as nn
from snntorch import surrogate
from snntorch import utils

def create_snn_network(input_size, num_classes):
    beta = 0.5  # Decay rate for the leaky integrate-and-fire neuron
    spike_grad = surrogate.fast_sigmoid()  # Surrogate gradient for backpropagation

    model = nn.Sequential(
        nn.Conv2d(input_size[0], 12, kernel_size=5, padding=2),
        nn.AvgPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad),
        nn.Conv2d(12, 32, kernel_size=5, padding=2),
        nn.AvgPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad),
        nn.Flatten(),
        nn.Linear(32 * (input_size[1] // 4) * (input_size[2] // 4), num_classes),
        snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)
    )
    return model

# Forward pass function
def forward_pass(net, data, device):
    spk_rec = []
    net.to(device)
    utils.reset(net)  # Resetting hidden states for all LIF neurons in the network

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, _ = net(data[step].to(device))
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)