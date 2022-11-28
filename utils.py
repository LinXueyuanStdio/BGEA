import torch


def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair


def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    rel_all = torch.cat([rel, rel + rel.max() + 1])
    return edge_index_all, rel_all


def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 5, 10)):
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    hits1 = get_hits_from_S(S, pair, dist, Hn_nums)
    return S, hits1


def get_hits_from_S(S, pair, dist='L1', Hn_nums=(1, 5, 10)):
    pair_num = pair.size(0)
    hits1 = None
    print('Left:\t', end='')
    for k in Hn_nums:
        pred_topk = S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item() / pair_num
        if k == 1:
            hits1 = Hk
        print('Hits@%d: %.2f%%    ' % (k, Hk * 100), end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1 / (rank + 1)).mean().item()
    print('MRR: %.3f' % MRR)
    print('Right:\t', end='')
    for k in Hn_nums:
        pred_topk = S.t().topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item() / pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk * 100), end='')
    rank = torch.where(S.t().sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1 / (rank + 1)).mean().item()
    print('MRR: %.3f' % MRR)
    return hits1


def get_S_from_embedding_list(x1, x2, pair):
    weight_list = [1] * len(x1)
    S_list = []
    for i in range(len(x1)):
        S = distS(x1[i], x2[i], pair).to("cuda:1")
        S_list.append(S)
    S = composeS(S_list, weight_list)
    return S


def distS(x1, x2, pair):
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    return S


def composeS(S_list, weight_list):
    assert len(S_list) == len(weight_list)
    assert len(S_list) >= 1
    S = None
    for i in range(len(S_list)):
        Si = S_list[i]
        wi = weight_list[i]
        if S is None:
            S = Si * wi
        else:
            S = S + Si * wi
    assert S is not None
    return S


def get_hits_stable_from_S(S, pair):
    pair_num = pair.size(0)
    # index = S.flatten().argsort(descending=True)
    index = (S.softmax(1) + S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index // pair_num
    index_e2 = index % pair_num
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num * 100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned / pair_num * 100))


def get_hits_stable(x1, x2, pair):
    pair_num = pair.size(0)
    S = -torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    # index = S.flatten().argsort(descending=True)
    index = (S.softmax(1) + S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index // pair_num
    index_e2 = index % pair_num
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num * 100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned / pair_num * 100))