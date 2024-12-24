"""
Taken from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
authors: TUM-LMF
"""
import torch.nn as nn
from torch.autograd import Variable
import torch
from STANet.models.base import BaseModel


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
        )


class ConvLSTM(BaseModel):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, pad_mask=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        pad_maks (b , t)
        Returns
        -------
        last_state_list, layer_output
        """
        # if not self.batch_first:
        #     # (t, b, c, h, w) -> (b, t, c, h, w)
        #     input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=input_tensor.size(0), device=input_tensor.device
            )

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            if pad_mask is not None:
                last_positions = (~pad_mask).sum(dim=1) - 1
                layer_output = layer_output[:, last_positions, :, :, :]

            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param





class BConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        super(BConvLSTM, self).__init__()
        self.convlstm_forward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.convlstm_backward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )

    def forward(self, input, pad_mask=None):
        # FORWARD
        _aa, forward_states = self.convlstm_forward(input, pad_mask=pad_mask)
        # out = forward_states[0][1]  # take last cell state as embedding
        out = _aa[0]  # take last cell state as embedding

        # BACKWARD
        x_reverse = torch.flip(input, dims=[1])
        if pad_mask is not None:
            pmr = torch.flip(pad_mask.float(), dims=[1]).bool()
            x_reverse = torch.masked_fill(x_reverse, pmr[:, :, None, None, None], 0)
            # Fill leading padded positions with 0s
        _bb, backward_states = self.convlstm_backward(x_reverse)

        out = torch.cat([out, _bb[0]], dim=1)
        return out


if __name__ == '__main__':

    model = ConvLSTM( input_size=(8,8), input_dim=1024, hidden_dim=1024, kernel_size=(3,3)).cuda()


    input = torch.rand(1, 16, 1024, 8, 8)
    input = input.to("cuda")
    print(model)
    output,b = model(input)
    print(output[0].shape,b[0][0].shape,b[0][1].shape)
    # print(output.shape)

    # convLSTM模型的layer_output和last_state在输出内容和用途上有一些区别。
    # layer_output是指ConvLSTM模型中每一层的输出结果列表，包含了每一层的特征表示或激活情况。这些输出可以用于分析网络内部的信息流动、特征提取情况以及进行可视化。
    # last_state是指ConvLSTM模型中每一层的最后一个状态，通常用于循环神经网络（RNN）或长短期记忆网络（LSTM）等模型中。这些状态包含了网络在处理序列数据时的最终状态信息，通常用于进一步的分析、预测或在某些任务中作为输出。
    # 因此，layer_output主要用于分析网络内部特征，而last_state主要用于获取模型在处理序列数据时的最终状态信息。
    #
    #
    # 在ConvLSTM模型中，C和H分别代表细胞状态（CellState）和隐藏状态（HiddenState），它们在LSTM中扮演着重要的角色。
    # C（细胞状态）是LSTM中的记忆单元，负责记忆和传递信息。它通过遗忘门、输入门和输出门的调控，控制着信息的流动和保存。C的更新受到遗忘门、输入门和输出门的调节，从而保留了长期的记忆。
    # H（隐藏状态）是LSTM在每个时间步输出的状态，也被称为输出状态。它是当前时间步的输出结果，包含了当前时间步的信息，并且会作为下一个时间步的输入。
    # 因此，C和H在ConvLSTM模型中代表着不同的信息和作用：C负责记忆和传递信息，H负责输出当前时间步的状态
    #
    # 在做分割任务时，通常会使用隐藏状态（H）来进行分割，因为隐藏状态包含了当前时间步的输出结果，是当前时间步的状态信息。隐藏状态可以提供有用的特征表示，有助于对图像或序列数据进行分割。细胞状态（C）通常用于传递和保存信息，而隐藏状态更适合用于任务的输出和特征提取。