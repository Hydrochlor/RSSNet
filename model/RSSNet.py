import torch
import torch.nn.functional as F
from torch import nn
import math


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """
    This class defines a normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: number of output channels
        """
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class FFN(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, max_length):
        pe = torch.zeros(max_length, in_channels)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, in_channels, 2, dtype=torch.float)
                * -(math.log(10000.0) / in_channels)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, n_head, dropout, is_casual):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_channels, 10000)
        self.attn_in_norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, n_head, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.is_casual = is_casual

    def forward(self, x):
        x = x.transpose(1, 2)
        attns = None
        output = self.pos_enc(self.attn_in_norm(x))
        output, _ = self.attn(output, output, output)
        output = self.norm(output + self.dropout(output))
        return output.transpose(1, 2)


class GA(nn.Module):
    def __init__(self, in_chan, out_chan, drop_path) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(out_chan, 8, 0.1, False)
        self.mlp = FFN(out_chan, out_chan * 2, drop=0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class LA(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out


class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = GA(
            in_channels * upsampling_depth, in_channels, 0.1
        )
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(LA(in_channels, in_channels, 5))

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # global features
        global_f = []
        for fea in output:
            global_f.append(F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            ))
        global_f = self.globalatt(torch.stack(global_f, dim=1).sum(1))  # [B, N, T]

        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            tmp = F.interpolate(global_f, size=output[idx].shape[-1], mode="nearest") + output[idx]
            x_fused.append(tmp)

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i + 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)

        return self.res_conv(expanded) + residual
    
class NetBlock(nn.Module):
    def __init__(self, out_channels, in_channels, upsampling_depth):
        super(NetBlock, self).__init__()
        self.intra_net = UConvBlock(out_channels = out_channels, in_channels = in_channels, upsampling_depth = upsampling_depth)
        self.inter_net = UConvBlock(out_channels = out_channels, in_channels = in_channels, upsampling_depth = upsampling_depth)

        self.intra_concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

        self.inter_concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )
        
    def forward(self, x):
        # x : [B, N, K, S]
        B, N, K, S = x.shape
        # intra : [BS, N, K]
        intra = x.permute(0, 3, 1, 2).contiguous().view(B*S, N, K)
        # intra : [BS, N, K]
        intra = self.intra_net(intra)
        # intra : [B, S, N, K]
        intra = intra.view(B, S, N, K)
        # intra : [B, N, K, S]
        intra = intra.permute(0, 2, 3, 1).contiguous()

        # intra : [B, N, K, S]
        intra = intra + self.intra_concat_block(x)

        # inter : [BK, N, S]
        inter = intra.permute(0, 2, 1, 3).contiguous().view(B*K, N, S)
        # inter : [BK, N, S]
        inter = self.inter_net(inter)
        # inter : [B, K, N, S]
        inter = inter.view(B, K, N, S)
        # inter : [B, N, K, S]
        inter = inter.permute(0, 2, 1, 3).contiguous()

        # out : [B, N, K, S]
        out = inter + self.inter_concat_block(intra)

        return out

class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, _iter=4):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net = NetBlock(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.net(x)
            else:
                x = self.net(self.concat_block(mixture + x))
        return x

class Net(nn.Module):
    def __init__(self, out_channels, enc_kernel_size, enc_num_basis, upsampling_depth = 4, iter = 4, K = 45, num_source = 2):
        super(Net, self).__init__()
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.out_channels = out_channels
        self.num_source = num_source
        self.K = K
        self.upsampling_depth = upsampling_depth
        self.iter = iter
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_basis,
            kernel_size=self.enc_kernel_size,
            stride=1,
            padding=self.enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=self.enc_num_basis,
            out_channels=self.out_channels,
            kernel_size=1)
        
        self.sm = Recurrent(out_channels = self.out_channels,
                            in_channels = self.enc_num_basis,
                            upsampling_depth = self.upsampling_depth,
                            _iter = self.iter)
        
        mask_conv = nn.Conv1d(self.out_channels, self.num_source * self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        self.mask_nl_class = nn.ReLU()
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_basis * self.num_source,
            out_channels=self.num_source,
            kernel_size=self.enc_kernel_size,
            stride=1,
            padding=self.enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap
    
    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        #print(input.shape)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap
    
    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        #print("gap:", gap)
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

    def forward(self, x):
        x = x.float()
        # input : [B, T]
        x = x.unsqueeze(1)
        # x : [B, 1, T]
        x = self.encoder(x)
        #print('after encoder:', x.shape)
        s = x.clone()

        x = self.ln(x)
        x = self.bottleneck(x)
        #print('after bottleneck:', x.shape)

        x, gap = self._Segmentation(x, self.K)

        #print('after segmentation:', x.shape)

        x = self.sm(x)

        #print('after sm:', x.shape)

        x = self._over_add(x, gap)

        #print('after overadd:', x.shape)
        
        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_source, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        #print('after masking:', x.shape)
        x = x * s.unsqueeze(1)

        x = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))

        return x
    
if __name__ == "__main__":
    net = Net(128, 3, 512, 4, 4, 45, 3)
    dummy_input = torch.rand(32, 1024)
    x = net(dummy_input)
    print(x.shape)