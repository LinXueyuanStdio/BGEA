import argparse
import itertools
from pathlib import Path

from data import DBP15K
from loss import L1_Loss
from model import *
from toolbox.DataSchema import read_cache
from toolbox.RandomSeeds import set_seeds
from utils import composeS, add_inverse_rels, get_hits, get_hits_stable, load_alignment_pair

set_seeds(2222)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--model_parallel", action="store_true", default=True)
    parser.add_argument("--GPU", type=list, nargs='+', default=[['0'], ['1']])
    parser.add_argument("--data", default="data")
    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--rate", type=float, default=0.3)

    parser.add_argument("--r_hidden", type=int, default=300)

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)

    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--neg_epoch", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=100)
    parser.add_argument("--reset_epoch", type=int, default=20)
    parser.add_argument("--stable_test", action="store_true", default=True)

    parser.add_argument("--keep_seeds", type=int, default=4500)
    parser.add_argument("--new_train_seeds", type=int, default=8000)
    parser.add_argument("--neg_seeds", type=int, default=2000)
    args = parser.parse_args()
    return args


args = parse_args()
if args.model_parallel == True:
    device = 'cuda:' + args.GPU[0][0]
    second_device = 'cuda:' + args.GPU[1][0]

else:
    device = 'cuda:' + args.GPU[0][0]
    second_device = 'cuda:' + args.GPU[0][0]

# load attribute similarity and value similarity
root = Path("data/%s/cache" % args.lang)
pairs = load_alignment_pair("data/%s/ref_ent_ids" % args.lang)
ratio = 0.3
test_pair = pairs[int(ratio * len(pairs)):]

test_seeds = torch.LongTensor(test_pair).to(second_device)

S1 = read_cache(root / "attr_similarity").to(second_device)
S2 = read_cache(root / "value_similarity").to(second_device)


def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    return x1, x2


def get_train_batch(x1, x2, train_set, k=5):
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k + 1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k + 1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k + 1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k + 1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    return train_batch


def train(model, criterion, optimizer, data, train_batch, false_pair=None):
    model.train()
    x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
    x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    loss = criterion(x1, x2, data.new_train_set, train_batch, false_pair)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def init_data(args, device):
    data = DBP15K(args.data, args.lang, rate=args.rate)[0]
    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)
    return data


def test(x1, x2, data, stable=False):
    with torch.no_grad():
        print('-' * 16 + 'Train_set' + '-' * 16)
        get_hits(x1, x2, data.new_train_set)
        print('-' * 16 + 'Test_set' + '-' * 17)
        S, hits1 = get_hits(x1, x2, data.test_set)
        if stable:
            get_hits_stable(x1, x2, data.test_set)
        print()
    return S.detach().cpu(), hits1


def generate_pairs(x1, x2, data):
    with torch.no_grad():
        seeds = data.test_set

        new_idx = []
        false_idx = []
        global_align = []

        S = torch.cdist(x1[seeds[:, 0]], x2[seeds[:, 1]], p=1)
        S = S.to(second_device)
        S = composeS([S1, S2, S], [0.5, 0.4, 0.1])
        S_global = -S

        pair_num = seeds.size(0)
        index = (S_global.softmax(1) + S_global.softmax(0)).flatten().argsort(descending=True)
        index_e1 = index // pair_num
        index_e2 = index % pair_num
        aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
        aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
        true_aligned = 0
        for _ in range(pair_num * 100):
            if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
                continue
            global_align.append((seeds[index_e1[_].item(), 0].item(), seeds[index_e2[_].item(), 1].item()))
            if index_e1[_] == index_e2[_]:
                true_aligned += 1
            aligned_e1[index_e1[_]] = True
            aligned_e2[index_e2[_]] = True

        left_pred = S.topk(1, largest=False)[1]
        right_pred = S.T.topk(1, largest=False)[1]

        positive = 0
        negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(left_pred.shape[0]):
            left_i = left_pred[i]
            j = right_pred[left_i]
            if i == j.item():
                new_idx.append((seeds[i, 0].item(), seeds[left_i.item(), 1].item()))
            else:
                negative += 1
                l2r_dist = S[i, left_i]
                r2l_dist = S.T[left_i, j]

                if l2r_dist > r2l_dist:
                    false_idx.append((seeds[i, 0].item(), seeds[left_i.item(), 1].item()))
                else:
                    false_idx.append((seeds[i, 0].item(), seeds[left_i.item(), 1].item()))

        new_idx = list(set(new_idx).intersection(set(global_align)))
        false_idx = list(set(false_idx).difference(set(global_align)))

        positive = len(new_idx)
        negative = len(false_idx)

        false_positive = 0
        false_negative = 0
        for p in new_idx:
            if p[0] != p[1]:
                false_positive += 1
        for f in false_idx:
            if f[0] == f[1]:
                false_negative += 1

        print()
        print("positive:", positive, "negative:", negative, "false_positive:", false_positive, "false_negative", false_negative)
        print("false_positive_rate:", false_positive / positive)
        print("false_negative_rate:", false_negative / negative)

        return new_idx, false_idx


def reset(x1, x2, data, keep_seeds=2000, new_train_seeds=7000, neg_seeds=1000):
    with torch.no_grad():
        new_pair, false_pair = generate_pairs(x1, x2, data)

        new_train_set = torch.LongTensor(new_pair)
        perm = torch.randperm(new_train_set.size(0))
        idx = perm[:keep_seeds]
        new_train_set = new_train_set[idx].to(second_device)

        perm = torch.randperm(data.train_set.size(0))
        idx = perm[:new_train_seeds]
        new_train_set_base = data.train_set[idx]
        data.new_train_set = torch.cat([new_train_set_base, new_train_set], dim=0).to(second_device)

        new_test_false_set = torch.LongTensor(false_pair)
        perm = torch.randperm(new_test_false_set.size(0))
        idx = perm[:neg_seeds]
        new_test_false_set = new_test_false_set[idx].to(second_device)
        return new_test_false_set


if __name__ == '__main__':

    data = init_data(args, second_device).to(second_device)

    model = EchoEA(device, second_device, data.x1.size(1), args.r_hidden)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x1, data.x2])))
    criterion = L1_Loss(args.gamma)
    data.new_train_set = data.train_set
    false_pair = None

    #     max_hits1 = 0
    #     save_epoch = 200

    for epoch in range(args.epoch):
        if epoch % args.neg_epoch == 0:
            x1, x2 = get_emb(model, data)
            train_batch = get_train_batch(x1, x2, data.new_train_set, args.k)
        loss = train(model, criterion, optimizer, data, train_batch, false_pair)
        print('Epoch:', epoch + 1, '/', args.epoch, '\tLoss: %.3f' % loss, '\r', end='')
        if (epoch + 1) % args.test_epoch == 0:
            print()
            S, hits1 = test(x1, x2, data, args.stable_test)

            # save model and embedding

        #             if(hits1 >= max_hits1 and epoch + 2 > save_epoch):
        #                 max_hits1 = hits1
        #                 cache_dir = Path("data/%s/cache/" % args.lang)
        #                 cache_dir.mkdir(exist_ok=True)
        #                 cache_data(S, cache_dir / ("rel_graph_similarity_e" + str(epoch+1) + "_n" + str(args.neg_epoch) + "_r" + str(args.reset_epoch)+"_hit"+str(int(hits1//1e-4))))
        #                 del S

        #                 torch.save([data.x1.clone().to('cpu'),data.x2.clone().to('cpu')], ("saved_models/"+ args.lang + "_emb_e"+  str(epoch+1) + "_n" + str(args.neg_epoch) + "_r" + str(args.reset_epoch)+"_hit"+str(int(hits1//1e-4))+".pt"))
        #                 torch.save(model.state_dict(),("saved_models/"+ args.lang + "_model_e"+  str(epoch+1) + "_n" + str(args.neg_epoch) + "_r" + str(args.reset_epoch)+"_hit"+str(int(hits1//1e-4))+".pt"))

        if (epoch + 1) % args.reset_epoch == 0:
            false_pair = reset(x1, x2, data,
                               keep_seeds=args.keep_seeds,
                               new_train_seeds=args.new_train_seeds,
                               neg_seeds=args.neg_seeds)
