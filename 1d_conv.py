import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv1d(9, 18, kernel_size=3)  # 9 input channels, 18 output channels
        self.conv2 = nn.Conv1d(18, 36, kernel_size=3)  # 18 input channels from previous Conv. layer, 36 out
        self.conv2_drop = nn.Dropout2d()  # dropout
        self.fc1 = nn.Linear(1044, 72)  # Fully-connected classifier layer
        self.fc2 = nn.Linear(72, 19)  # Fully-connected classifier layer

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        print('1', x.shape)
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        print('2', x.shape)

        # point A
        x = x.view(x.shape[0], -1)

        # point B
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

        # ##TODO WONDERIT conv1d
        # def conv1d(
        #         input,
        #         weight,
        #         bias=None,
        #         stride=1,
        #         padding=0,
        #         dilation=1,
        #         groups=1,
        #         padding_mode="zeros",
        # ):
        #     """
        #     Overloads torch.conv1d to be able to use MPC on convolutional networks.
        #     The idea is to build new tensors from input and weight to compute a
        #     matrix multiplication equivalent to the convolution.
        #
        #     Args:
        #         input: input image
        #         weight: convolution kernels
        #         bias: optional additive bias
        #         stride: stride of the convolution kernels
        #         padding:
        #         dilation: spacing between kernel elements
        #         groups:
        #         padding_mode: type of padding, should be either 'zeros' or 'circular' but 'reflect' and 'replicate' accepted
        #     Returns:
        #         the result of the convolution as an AdditiveSharingTensor
        #     """
        #     # Currently, kwargs are not unwrapped by hook_args
        #     # So this needs to be done manually
        #     if bias.is_wrapper:
        #         bias = bias.child
        #
        #     assert len(input.shape) == 3
        #     assert len(weight.shape) == 3
        #
        #     # Change to tuple if not one
        #     stride = torch.nn.modules.utils._single(stride)
        #     padding = torch.nn.modules.utils._single(padding)
        #     dilation = torch.nn.modules.utils._single(dilation)
        #
        #     # Extract a few useful values
        #     batch_size, nb_channels_in, nb_cols_in = input.shape
        #     nb_channels_out, nb_channels_kernel, nb_cols_kernel = weight.shape
        #
        #     print(batch_size, nb_channels_in, nb_cols_in)
        #     print(nb_channels_out, nb_channels_kernel, nb_cols_kernel)
        #
        #     if bias is not None:
        #         assert len(bias) == nb_channels_out
        #
        #     # Check if inputs are coherent
        #     assert nb_channels_in == nb_channels_kernel * groups
        #     assert nb_channels_in % groups == 0
        #     assert nb_channels_out % groups == 0
        #
        #     # Compute output shape
        #     # nb_rows_out = int(
        #     #     ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0])
        #     #     + 1
        #     # )
        #     nb_cols_out = int(
        #         ((nb_cols_in + 2 * padding[0] - dilation[0] * (nb_cols_kernel - 1) - 1) / stride[0])
        #         + 1
        #     )
        #
        #     # Apply padding to the input
        #     if padding != (0, 0):
        #         padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        #         input = torch.nn.functional.pad(
        #             input, (padding[0], padding[0]), padding_mode
        #         )
        #         # Update shape after padding
        #         # nb_rows_in += 2 * padding[0]
        #         nb_cols_in += 2 * padding[0]
        #
        #     # We want to get relative positions of values in the input tensor that are used by one filter convolution.
        #     # It basically is the position of the values used for the top left convolution.
        #     pattern_ind = []
        #     for ch in range(nb_channels_in):
        #         # for r in range(nb_rows_kernel):
        #         for c in range(nb_cols_kernel):
        #             pixel = c * dilation[0]
        #             pattern_ind.append(pixel + ch * nb_cols_in)
        #
        #     # The image tensor is reshaped for the matrix multiplication:
        #     # on each row of the new tensor will be the input values used for each filter convolution
        #     # We will get a matrix [[in values to compute out value 0],
        #     #                       [in values to compute out value 1],
        #     #                       ...
        #     #                       [in values to compute out value nb_rows_out*nb_cols_out]]
        #     im_flat = input.view(batch_size, -1)
        #     im_reshaped = []
        #     # for cur_row_out in range(nb_rows_out):
        #     for cur_col_out in range(nb_cols_out):
        #         # For each new output value, we just need to shift the receptive field
        #         offset = cur_col_out * stride[0]
        #         tmp = [ind + offset for ind in pattern_ind]
        #         im_reshaped.append(im_flat[:, tmp])
        #
        #     im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)
        #     # The convolution kernels are also reshaped for the matrix multiplication
        #     # We will get a matrix [[weights for out channel 0],
        #     #                       [weights for out channel 1],
        #     #                       ...
        #     #                       [weights for out channel nb_channels_out]].TRANSPOSE()
        #     weight_reshaped = weight.view(nb_channels_out // groups, -1).t()
        #
        #     # Now that everything is set up, we can compute the result
        #     if groups > 1:
        #         res = []
        #         chunks_im = torch.chunk(im_reshaped, groups, dim=1)
        #         chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
        #         for g in range(groups):
        #             tmp = chunks_im[g].matmul(chunks_weights[g])
        #             res.append(tmp)
        #         res = torch.cat(res, dim=1)
        #     else:
        #         res = im_reshaped.matmul(weight_reshaped)
        #
        #     # Add a bias if needed
        #     if bias is not None:
        #         res += bias
        #
        #     # ... And reshape it back to an image
        #     res = (
        #         res.permute(0, 2, 1)
        #             .view(batch_size, nb_channels_out, nb_cols_out)
        #             .contiguous()
        #     )
        #     return res
        #
        # module.conv1d = conv1d

if __name__ == '__main__':
    m = model()
    data = torch.randn(64, 9, 125)
    print('data', data.shape)
    out = m(data)
    print('main', out.shape)

