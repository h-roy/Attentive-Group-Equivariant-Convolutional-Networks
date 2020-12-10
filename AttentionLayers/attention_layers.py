''''
################################################################################
############## Reimplementing the attention layers #############################
################################################################################
'''
import torch
import math
import numpy as np
from AttentionLayers import ConvGGLayer
from AttentionLayers import ConvRnGLayer

'''
Channel Attention function for the lifting layer
'''

class ChannelAttention(torch.nn.Module):
    def __init__(self, N_out, N_in, ratio=1):
        self.N_in = N_in
        self.N_out = N_out
        self.ratio = ratio

    def forward(self, input):

        self.weight_1 = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in // ratio, self.N_in))
        self.weight_2 = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, self.N_in // ratio))
        torch.nn.init.normal_(self.weight_1)
        torch.nn.init.normal_(self.weight_1)
        spatial_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        spatial_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        mean_att = ((spatial_mean.unsqueeze(-3)) * (self.weight_1.reshape(1, self.weight_1.shape[0], 1, self.weight_1.shape[1], self.weight_1.shape[2], 1))).sum(-2)
        mean_att = (((torch.relu(mean_att)).unsqueeze(-3)) * (self.weight_2.reshape(1, self.weight_2.shape[0], 1, self.weight_2.shape[1], self.weight_2.shape[2], 1))).sum(-2)
        max_att = ((spatial_max.unsqueeze(-3)) * (self.weight_1.reshape(1, self.weight_1.shape[0], 1, self.weight_1.shape[1], self.weight_1.shape[2], 1))).sum(-2)
        max_att = (((torch.relu(max_att)).unsqueeze(-3)) * (self.weight_2.reshape(1, self.weight_2.shape[0], 1, self.weight_2.shape[1], self.weight_2.shape[2], 1)))).sum(-2)
        output = torch.sigmoid(max_att + mean_att)
        output = output.reshape([input.shape[0], self.N_out, input.shape[2], self.N_in, 1, 1])
        return output


'''
Channel Attention function for the standard layer
'''

class ChannelAttentionGG(torch.nn.Module):
    def __init__(self, N_h, N_out, N_h_in, N_in, ratio=1):

        self.N_in = N_in
        self.N_out = N_out
        self.ratio = ratio
        self.N_h_in = N_h_in
        self.N_h = N_h

    def forward(self, input):

        self.weight_1 = torch.nn.Parameter(torch.rand(self.N_out, self.N_in // ratio, self.N_in, self.N_h_in))
        self.weight_2 = torch.nn.Parameter(torch.rand(self.N_out, self.N_in, self.N_in // ratio, self.N_h_in))
        NN_layer_1_list = []
        NN_layer_2_list = []
        for i in range(self.N_h):
            NN_layer_1_list.append(self.weight_1.roll(shifts=i, dims=-1))
            NN_layer_2_list.append(self.weight_2.roll(shifts=i, dims=-1))
        layer_1 = torch.stack(NN_layer_1_list, dim=1)
        layer_2 = torch.stack(NN_layer_2_list, dim=1)
        spatial_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        spatial_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        mean_att = (((spatial_mean).unsqueeze(-4)) * (layer_1.reshape([1, layer_1.shape[0], layer_1.shape[1], layer_1.shape[2], layer_1.shape[3], layer_1.shape[4], 1]))).sum(-3)
        mean_att = (((torch.relu(mean_att)).unsqueeze(-4)) * (layer_2.reshape([1, layer_2.shape[0], layer_2.shape[1], layer_2.shape[2], layer_2.shape[3], layer_2.shape[4], 1]))).sum(-3)
        max_att = (((spatial_max).unsqueeze(-4)) * (layer_1.reshape([1, layer_1.shape[0], layer_1.shape[1], layer_1.shape[2], layer_1.shape[3], layer_1.shape[4], 1]))).sum(-3)
        max_att = (((torch.relu(max_att)).unsqueeze(-4)) * (layer_2.reshape([1, layer_2.shape[0], layer_2.shape[1], layer_2.shape[2], layer_2.shape[3], layer_2.shape[4], 1]))).sum(-3)
        output = torch.sigmoid(max_att + mean_att)
        output = output.reshape([input.shape[0], self.N_out, self.N_h, -1, self.N_h_in, 1, 1])
        return output

'''
Spatial Attention function for the lifting layer
'''

class SpatialAttention(torch.nn.Module):
    def __init__(self,group,N_in,N_out,kernel_size,h_grid,stride,dilation=1,wscale=1.0):


        self.group = group
        self.H = group.H
        N_in = 2
        self.N_in = N_in
        self.N_out = N_out
        self.kernel_size = kernel_size
        self.h_grid = h_grid
        self.stride = stride
        self.dilation = dilation
        self.wscale = wscale
        self.con_groups = len(h_grid.grid)
        self.padding = dilation * (kernel_size //2)
        n = self.N_in
        k = self.kernel_size ** 2
        n *= k
        stdv = wscale * (1. / math.sqrt(n))
        self.weight = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, input):

        channel_mean = torch.mean(input, dim=-3, keepdim=True)
        channel_max = torch.max(input, dim=-3, keepdim=True)
        attention_input = torch.cat([channel_mean, channel_max], dim=-3)
        kernel_list = []
        for h in self.h_grid.grid:
            kernel_list.append(self.H.left_representation_on_Rn(h, self.weight))
        kernel = torch.stack(kernel_list, dim=1)
        kernel = kernel.reshape([self.N_out * self.N_h * self.N_in, 1, kernel_stack.shape[-2], kernel_stack.shape[-1]])
        attention_input = attention_input.reshape([attention_input.shape[0], self.N_out * self.N_h * self.N_in, attention_input.shape[-2], attention_input.shape[-1]])
        output = torch.conv2d(input=attention_input,weight=kernel,bias=None,stride=self.stride,adding=self.padding,dilation=self.dilation,groups=self.N_out*self.N_h*self.N_in)
        output = torch.sigmoid((output.reshape([output.shape[0], self.N_out, self.N_h, self.N_in, output.shape[-2], output.shape[-1]])).sum(-3, keepdim=True))
        return output

'''
Spatial Attention function for the standard layer
'''

class SpatialAttentionGG(ConvGGLayer):
    def __init__(self,group,N_in,N_out,kernel_size,h_grid,input_h_grid,stride,dilation=1,wscale=1.0):

        N_in = 2
        self.group = group
        self.H = group.H
        N_in = 2
        self.N_in = N_in
        self.N_out = N_out
        self.kernel_size = kernel_size
        self.h_grid = h_grid
        self.input_h_grid = input_h_grid
        self.stride = stride
        self.dilation = dilation
        self.wscale = wscale
        self.con_groups = len(h_grid.grid)
        self.padding = dilation * (kernel_size //2)
        n = self.N_in
        k = self.kernel_size ** 2
        n *= k
        stdv = wscale * (1. / math.sqrt(n))
        self.weight = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        channel_mean = torch.mean(input, dim=-4, keepdim=True)
        channel_max = torch.max(input, dim=-4, keepdim=True)
        attention_input = torch.cat([channel_mean, channel_max], dim=-4)
        kernel_list = []
        for h in self.h_grid.grid:
            kernel_list.append(self.H.left_representation_on_Rn(h, self.weight))
        kernel = torch.stack(kernel_list, dim=1)
        kernel = kernel.reshape([self.N_h * self.N_out * self.N_in * self.N_h_in, 1, self.kernel_size, self.kernel_size])
        attention_input = attention_input.reshape([attention_input.shape[0], self.N_h * self.N_out * self.N_in * self.N_h_in, attention_input.shape[-2], attention_input.shape[-1]])
        output = torch.conv2d(input=attention_input,weight=kernel,bias=None,stride=self.stride,adding=self.padding,dilation=self.dilation,groups=self.N_h * self.N_h_in * self.N_out * self.N_in)
        output = torch.sigmoid((output.reshape([input.shape[0], self.N_out, self.N_h, self.N_in, self.N_h_in, input.shape[-2], input.shape[-1]])).sum(-4, keepdim=True))
        return output

''''
################################################################################
############## End of attention layers #########################################
################################################################################
'''

################################################################################
##################### From Romero (2020)########################################
################################################################################


''''
TODO: reimplement feature attetntion functions!
'''

'''
As explained in the paper, due to the required storage of all individual channelwise convolutions, the ammount of
memory required is too big for current hardware architectures ( in CIFAR10 requires ~ 72GBs of GPU memory).
Resutantly, we implement FeatureMapsAttention. Here, attention is applied directly to the feature maps and subsequently
passed to the convolution. The underlying implication is that it is assumed that the attention maps are independent
of the N_outs and hence, one such computations is sufficient for all the feature representations produced subsequently
by the convolution.
'''
##########################################################################
############################ FeatMapsAttention ###########################
##########################################################################
class fChannelAttention(torch.nn.Module):
    def __init__(self, N_in, ratio=1):
        super(fChannelAttention, self).__init__()
        self.N_in = N_in
        self.ratio = ratio
        self.weight_fc1 = torch.nn.Parameter(torch.Tensor(self.N_in // ratio, self.N_in))
        self.weight_fc2 = torch.nn.Parameter(torch.Tensor(self.N_in, self.N_in // ratio))
        # TODO Include bias
        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # Default initialization in torch
        torch.nn.init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight_fc2, a=math.sqrt(5))

    def forward(self, input):
        # Get statistics
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)             # [B, N_in, 1]
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)   # [B, N_in, 1]
        # Pass to layers
        avg_out = self._linear(torch.relu(self._linear(input_mean, self.weight_fc1)), self.weight_fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, self.weight_fc1)), self.weight_fc2)
        out = torch.sigmoid(avg_out + max_out)
        # Reshape output as input
        out = torch.reshape(out, [input.shape[0], self.N_in, 1, 1])
        # Return output
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-3)                       # [B, N_linear_out, N_in, 1]
        w_reshaped = w.reshape(1, w.shape[0], w.shape[1], 1)    # [B, N_linear_out, N_in, 1]
        output = (in_reshaped * w_reshaped).sum(-2)
        return output


class fChannelAttentionGG(torch.nn.Module):
    def __init__(self, N_h_in, N_in, ratio=1, group='SE2'):
        super(fChannelAttentionGG, self).__init__()
        self.N_in = N_in
        self.ratio = ratio
        self.N_h_in = N_h_in
        self.N_h = N_h_in
        self.weight_fc1 = torch.nn.Parameter(torch.rand(self.N_in // ratio, self.N_in, self.N_h_in))
        self.weight_fc2 = torch.nn.Parameter(torch.rand(self.N_in, self.N_in // ratio, self.N_h_in))

        # group instantiation
        self.action = self._left_action_of_h_grid_se2
        if group == 'E2':       # TODO: Move to a more meaningful place.
            import importlib
            group = importlib.import_module('attgconv.group.' + group)
            import attgconv
            e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
            # Create H grid for p4m group
            n_grid = 8
            self.h_grid = e2_layers.H.grid_global(n_grid)
            self.action = self._left_action_on_grid_e2

        # Initialize weights
        self.reset_parameters()
        # TODO Include bias

    def reset_parameters(self):
        # Default initialization in torch
        torch.nn.init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight_fc2, a=math.sqrt(5))

    # It is important to notice that this is equivalent to a 1x1 group convolution.
    def forward(self, input):
        # Create stack of filters
        fc1, fc2 = self.action()                                    # [N_lin_out, N_h, N_in, N_h_in]
        # Get statistics
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)                         # [B, N_in, N_h_in, 1]
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)               # [B, N_in, N_h_in, 1]
        # Pass to layers
        avg_out = self._linear(torch.relu(self._linear(input_mean, fc1)), fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, fc1)), fc2)
        out = torch.sigmoid(avg_out + max_out)
        # Reshape output as input
        out = torch.reshape(out, [out.shape[0], self.N_in, self.N_h_in, 1, 1])      # [B, N_in, N_h_in, 1, 1]
        # Return output
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-4).unsqueeze(-5)                                         # [B, N_linear_out, N_h, N_in, N_h_in, 1]
        w_reshaped = torch.reshape(w, [1, w.shape[0], w.shape[1], w.shape[2], w.shape[3], 1])   # [B, N_linear_out, N_h, N_in, N_h_in, 1]
        output = (in_reshaped * w_reshaped).sum(dim=[-3,-2])                                    # [B, N_linear_out, N_h, 1]
        return output

    def _left_action_of_h_grid_se2(self):   #TODO: Maybe it would be nice to move it to the group def itself.
        fc1 = torch.stack([self.weight_fc1.roll(shifts=i, dims=-1) for i in range(self.N_h)], dim=1)
        fc2 = torch.stack([self.weight_fc2.roll(shifts=i, dims=-1) for i in range(self.N_h)], dim=1)
        return fc1, fc2

    def _left_action_on_grid_e2(self):   #TODO: Maybe it would be nice to move it to the group def itself.
        fc1 = torch.stack([self._left_action_of_h_grid_e2(h, self.weight_fc1) for h in self.h_grid.grid], dim=1)
        fc2 = torch.stack([self._left_action_of_h_grid_e2(h, self.weight_fc2) for h in self.h_grid.grid], dim=1)
        return fc1, fc2

    def _left_action_of_h_grid_e2(self, h, fx):
        # They rotate in opposite directions
        shape = fx.shape
        Lgfx = fx.clone()
        # Now permute the axes
        Lgfx = torch.reshape(Lgfx, [shape[0], shape[1], 2, 4])
        # First permutation on rotate, then on mirror
        if h[0] != 0:
            Lgfx[:, :, 0, :] = torch.roll(Lgfx[:, :, 0, :], shifts=int(torch.round((1. / (np.pi / 2.) * h[0])).item()), dims=-1)
            Lgfx[:, :, 1, :] = torch.roll(Lgfx[:, :, 1, :], shifts=-int(torch.round((1. / (np.pi / 2.) * h[0])).item()), dims=-1)
        if h[-1] == -1:
            # Then on the m axis
            Lgfx = torch.roll(Lgfx, shifts=1, dims=-2)
        # Reshape
        Lgfx = torch.reshape(Lgfx, shape)
        # Return Lgfx
        return Lgfx


# In this case, it corresponds to a traditional convolution following a similar methodology to that of CBAM (Park et. al. 2019).
# However, if we apply conventional convolutions, we break equivariance to Rn \rtimes H. Instead of this, we utilize an ConvRnG ocnvolutions
# and perform avg_pooling next.
class fSpatialAttention(ConvRnGLayer):
    def __init__(self,
                 group,
                 kernel_size,
                 h_grid,
                 stride=1,
                 dilation=1,
                 groups=1,
                 wscale=1.0
                 ):

        # Set parameters and save in self
        N_in = 2            #  TODO Needs to be replaced for number of statistics.
        N_out = 1           #  One channel that describes attention spatially.
        padding = dilation * (kernel_size // 2)
        # ------------------------------
        super(fSpatialAttention, self).__init__(group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, groups, wscale)

        # self.N_in = N_in
        # self.N_out = N_out
        # self.stride = stride
        # self.kernel_size = kernel_size
        # self.padding = (kernel_size // 2)
        # self.dilation = dilation
        # self.groups = groups
        # self.wscale = wscale
        # self.weight = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size))
        # # Initialize
        # self._reset_parameters(wscale)

    # Method overriding:
    def forward(self, input, visualize=False):
        return self.f_att_conv2d(input, visualize)

    def f_att_conv2d(self, input, visualize):
        # Get statistics
        avg_in = torch.mean(input, dim=-3, keepdim=True)
        max_in, _ = torch.max(input, dim=-3, keepdim=True)
        input = torch.cat([avg_in, max_in], dim=-3)
        # Apply convolution
        output = self.conv_Rn_G(input)
        # Do pooling over the group
        output, _ = output.max(dim=2)
        # Apply sigmoid
        output = torch.sigmoid(output)
        # Visualization
        if False:
            self.att_map = output
        # Return output
        return output


# Corresponds to G-convolution followed by a normalization (sigmoid) step.
from attgconv import ConvGGLayer
class fSpatialAttentionGG(ConvGGLayer):
    def __init__(self,
                 group,
                 kernel_size,
                 input_h_grid,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 wscale=1.0
                 ):
        N_in = 2                 #TODO Needs to be replaced for number of statistics.
        N_out = 1
        padding = dilation * (kernel_size // 2)
        super(fSpatialAttentionGG, self).__init__(group, N_in, N_out, kernel_size, input_h_grid, input_h_grid, stride, padding, dilation, groups, wscale)

    # Method overriding:
    def forward(self, input, visualize=False):
        return self.f_att_conv_GG(input, visualize)

    def f_att_conv_GG(self, input, visualize):
        # Get input statistics
        avg_in = torch.mean(input, dim=-4, keepdim=True)
        max_in, _ = torch.max(input, dim=-4, keepdim=True)
        input = torch.cat([avg_in, max_in], dim=-4)
        # Apply group convolution
        output = self.conv_G_G(input)
        # Apply sigmoid
        output = torch.sigmoid(output)
        # Visualization
        if False:
            self.att_map = output
        # Return the output
        return output
