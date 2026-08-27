import torch
from itertools import permutations
import torch.nn.functional as F

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def mse(x, s):
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate mse, {} vs {}".format(
                x.shape, s.shape))
    x_zm = F.normalize(x, float('inf'), dim = -1)
    s_zm = F.normalize(s, float('inf'), dim = -1)
    t = torch.sum((x_zm - s_zm) * (x_zm - s_zm), dim = -1)
    
    return t

def Lossmse(ests, egs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def mse_loss(permute):
        # for one permute
        return sum(
            [mse(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [mse_loss(p) for p in permutations(range(num_spks))])
    
    max_perutt, _ = torch.min(sisnr_mat, dim=0)
    # si-snr
    return torch.sum(max_perutt) / N
    

def Loss(ests, egs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)
    
    ests = torch.transpose(ests, 0, 1)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def Loss_notrans(ests, egs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)
    
    #ests = torch.transpose(ests, 0, 1)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def Loss_SISNRi(ests, egs, mix):
    # spks x n x S
    refs = egs
    num_spks = len(refs)
    ests = torch.transpose(ests, 0, 1)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])

    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    return torch.sum(max_perutt - sum([sisnr(mix, refs[i]) for i in range(num_spks)]) / num_spks) / N

def Loss_SISNRi_notrans(ests, egs, mix):
    # spks x n x S
    refs = egs
    num_spks = len(refs)
    #ests = torch.transpose(ests, 0, 1)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])

    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    return torch.sum(max_perutt - sum([sisnr(mix, refs[i]) for i in range(num_spks)]) / num_spks) / N

def segoverlapsisnr(x, s, chunklen, gamma):
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    seglen = chunklen // 2
    seg_x_list = torch.split(x, seglen, dim=-1)
    seg_s_list = torch.split(s, seglen, dim=-1)
    x_list = []
    s_list = []
    for i in range(len(seg_x_list) - 1):
        x_list.append(torch.cat((seg_x_list[i], seg_x_list[i+1]), dim = -1))
        s_list.append(torch.cat((seg_s_list[i], seg_s_list[i+1]), dim = -1))
    sisnr_list = torch.stack([sisnr(x_list[i], s_list[i]) for i in range(len(x_list))], dim=1)
    avgsisnr = torch.mean(sisnr_list, dim=-1)
    return avgsisnr

def segfocalsisnr(x, s, chunklen, gamma, eps=1e-8):
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_list = torch.split(x, chunklen, dim=-1)
    s_list = torch.split(s, chunklen, dim=-1)
    sisnr_list = torch.stack([sisnr(x_list[i], s_list[i], eps) for i in range(len(x_list))], dim=1)

    #print('list:', sisnr_list)

    # weight = torch.stack([torch.softmax(sisnr_list[i], dim = 0) for i in range(len(sisnr_list))])
    
    # tmp = torch.pow(1 - weight, gamma) * sisnr_list
    # res = torch.mean(tmp, dim = -1)
    res = torch.mean(sisnr_list, dim = -1)
    return res

def focalLoss(ests, egs, chunklen, gamma):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [segoverlapsisnr(ests[s], refs[t], chunklen, gamma)
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def Loss2(ests, egs, alpha = 1, beta = 0.25):
    #ests: spks x spks x n x S
    #egs: spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(e, r, permute):
        # for one permute
        return sum(
            [sisnr(e[s], r[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    N = egs[0].size(0)
    max_perutt = []
    for i in range(num_spks):
        sisnr_mat = torch.stack([sisnr_loss(ests[i], refs, p) for p in permutations(range(num_spks))])
        max_perutt.append(torch.max(sisnr_mat, dim=0)[0])
    tmp = torch.stack([(-torch.sum(max_perutt[i]) / N) for i in range(num_spks)])
    loss1 = torch.min(tmp)

    sisnr_mat = torch.stack([sisnr_loss(ests[0], ests[1], p) for p in permutations(range(num_spks))])
    max_p, _ = torch.max(sisnr_mat, dim=0)
    loss2 = -torch.sum(max_p) / N
    loss = alpha * loss1 + beta * loss2

    return loss, loss1, loss2

def Loss3(ests, egs, factors, mixs, alpha = 1, beta = 0.25, gamma = 0.05, limit = -15):
    #ests: spks x spks x n x S
    #egs: spks x n x S
    #factors: spks x n x 1
    #mixs: n x spks x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(e, r, permute):
        # for one permute
        return sum(
            [sisnr(e[s], r[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    N = egs[0].size(0)
    max_perutt = []
    for i in range(num_spks):
        sisnr_mat = torch.stack([sisnr_loss(ests[i], refs, p) for p in permutations(range(num_spks))])
        max_perutt.append(torch.max(sisnr_mat, dim=0)[0])
    tmp = torch.stack([(-torch.sum(max_perutt[i]) / N) for i in range(num_spks)])
    loss1 = torch.min(tmp)

    sisnr_mat = torch.stack([sisnr_loss(ests[0], ests[1], p) for p in permutations(range(num_spks))])
    max_p, _ = torch.max(sisnr_mat, dim=0)
    loss2 = -torch.sum(max_p) / N

    SNRs = []
    # est_mix1 = ests[0][0] * factors[0] + ests[0][1] * (1 - factors[0])
    # est_mix2 = ests[0][1] * factors[0] + ests[0][0] * (1 - factors[0])
    # snr1 = torch.sum(sisnr(est_mix1, mixs[:, 0])) / N
    # snr2 = torch.sum(sisnr(est_mix2, mixs[:, 0])) / N
    # print(snr1)
    # print(snr2)
    for i in range(N):
        est_mix1 = ests[0][0][i] * factors[0][i].item() + ests[0][1][i] * (1 - factors[0][i].item())
        est_mix2 = ests[0][1][i] * factors[0][i].item() + ests[0][0][i] * (1 - factors[0][i].item())
        tmp = torch.stack([sisnr(est_mix1, mixs[i][0]), sisnr(est_mix2, mixs[i][0])])
        snr1 = torch.max(tmp)
        est_mix3 = ests[1][0][i] * factors[1][i].item() + ests[1][1][i] * (1 - factors[1][i].item())
        est_mix4 = ests[1][1][i] * factors[1][i].item() + ests[1][0][i] * (1 - factors[1][i].item())
        tmp = torch.stack([sisnr(est_mix3, mixs[i][1]), sisnr(est_mix4, mixs[i][1])])
        snr2 = torch.max(tmp)
        SNRs.append((snr1 + snr2) / 2)
    SNR_mat = torch.stack(SNRs)
    loss3 = -torch.sum(SNR_mat) / N
    
    if loss1 > limit:
        loss = alpha * loss1 + beta * loss2
    else:
        loss = alpha * loss1 + beta * loss2 + gamma * loss3

    return loss, loss1, loss2, loss3

def SI_SNR(_s, s, zero_mean=True):
    '''
         Calculate the SNR indicator between the two audios. 
         The larger the value, the better the separation.
         input:
               _s: Generated audio
               s:  Ground Truth audio
         output:
               SNR value 
    '''
    if zero_mean:
        _s = _s - torch.mean(_s)
        s = s - torch.mean(s)
    s_target = sum(torch.mul(_s, s))*s/torch.pow(torch.norm(s, p=2), 2)
    e_noise = _s - s_target
    return 20*torch.log10(torch.norm(s_target, p=2)/torch.norm(e_noise, p=2))

if __name__ == "__main__":
    # a = torch.rand((2, 2, 8, 1024))
    # b = torch.rand((2, 8, 1024))
    # mixs = torch.rand(8, 2, 1024)
    # factors = [torch.rand(8, 1), torch.rand(8, 1)]
    # #print(Loss2(a,b)[0].item())
    # l, l1, l2, l3 = Loss3(a, b, factors, mixs)
    # print(l, l1, l2, l3)
    # for p in permutations(range(2)):
    #     for s, t in enumerate(p):
    #         print(s, t)
    # a = torch.rand((32, 2, 1024))
    # b = torch.rand((2, 32, 1024))
    # c = torch.rand((32, 1024))
    # print(Loss_SISNRi(a, b, c))
    for p in permutations(range(2)):
        print(p)
    #print(Loss(a,b))