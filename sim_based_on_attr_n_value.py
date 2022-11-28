import os
from pathlib import Path

import numpy as np
import torch

from toolbox.DataSchema import cache_data
from toolbox.Progbar import Progbar
from utils import load_alignment_pair, get_hits_from_S


def get_attr_name2id_dict(path):
    att_dict = {}
    with open(path, "r", encoding='utf-8') as file:
        for line in file:
            att = line.split()[1]
            index = line.split()[0]
            att_dict[att] = index
    return att_dict


def get_attr_name2id_dicts(index_att_1, index_att_2):
    dict1 = get_attr_name2id_dict(index_att_1)
    dict2 = get_attr_name2id_dict(index_att_2)
    return dict1, dict2


def jaccard_similarity(source_set, target_set):
    if len(source_set) == 0 or len(target_set) == 0:
        return 0
    return len(source_set.intersection(target_set)) / len(source_set.union(target_set))


def preprocess_value(value):
    final_val = set()
    for string in value:
        word_set = set(string.split('_'))
        final_val.update(word_set)
    return final_val


def get_the_raw_datastructure(path, att_dict_inverse, index_keeped=None):
    data = {}
    seen_id = set()
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data_line = line.split()
            id = data_line[0]
            if not id in seen_id:
                data[id] = {
                    'att': set(),
                    'att_value': {}
                }
                seen_id.add(id)
            att_index = data_line[1]
            try:
                att_index = att_dict_inverse[att_index]
            except:
                continue
            if index_keeped:
                if att_index not in index_keeped:
                    continue
            data[id]['att'].add(att_index)
            data[id]['att_value'][att_index] = preprocess_value(data_line[2:])
    return data


def get_simi_att(lang="zh_en", ratio=0.3):
    dt_name_path = "data/%s/" % lang
    ent_att_val1 = os.path.join(dt_name_path, "ent_att_val_1")
    ent_att_val2 = os.path.join(dt_name_path, "ent_att_val_2")
    index_att_1 = os.path.join(dt_name_path, "index_att_1")
    index_att_2 = os.path.join(dt_name_path, "index_att_2")
    pairs = load_alignment_pair(dt_name_path + "ref_ent_ids")
    test_pair = pairs[int(ratio * len(pairs)):]

    att_dict1, att_dict2 = get_attr_name2id_dicts(index_att_1, index_att_2)
    source_att_set = set(att_dict1.keys())
    target_att_set = set(att_dict2.keys())
    kept_att = source_att_set.intersection(target_att_set)
    att_dict_inverse1 = {v: k for k, v in att_dict1.items()}
    att_dict_inverse2 = {v: k for k, v in att_dict2.items()}
    s_data = get_the_raw_datastructure(ent_att_val1, att_dict_inverse1, kept_att)
    t_data = get_the_raw_datastructure(ent_att_val2, att_dict_inverse2, kept_att)
    Ns = len(test_pair)
    Nt = len(test_pair)

    attr_similarity = np.zeros((Ns, Nt))
    value_similarity = np.zeros((Ns, Nt))
    print("computing attr and value similarity for %s" % lang)
    progbar = Progbar(max_step=Ns)
    for si in range(Ns):
        s = str(test_pair[si][0])
        progbar.update(si + 1, [("node", s)])
        if (s not in s_data) or (s in s_data and len(s_data[s]['att']) == 0):
            continue
        s_att_set = s_data[s]['att']
        for ti in range(Nt):
            t = str(test_pair[ti][1])
            if (t not in t_data) or (t in t_data and len(t_data[t]['att']) == 0):
                continue
            t_att_set = t_data[t]['att']
            common_att_set = s_att_set.intersection(t_att_set)
            value_similar = 0
            for ele in common_att_set:
                s_value_set = s_data[s]['att_value'][ele]
                t_value_set = t_data[t]['att_value'][ele]
                value_similar += jaccard_similarity(s_value_set, t_value_set)
            if value_similar > 0:
                value_similar = value_similar / len(common_att_set)
            value_similarity[si, ti] = value_similar
            attr_similarity[si, ti] = jaccard_similarity(s_att_set, t_att_set)
    return attr_similarity, value_similarity, test_pair


if __name__ == '__main__':
    dataset = ["zh_en", "fr_en", "ja_en"]
    for i in dataset:
        attr_similarity, value_similarity, test_pair = get_simi_att(i)
        attr_similarity = -torch.Tensor(attr_similarity)
        value_similarity = -torch.Tensor(value_similarity)
        print(i)
        print("attribute similarity")
        get_hits_from_S(attr_similarity, torch.LongTensor(test_pair))
        print("attribute similarity")
        get_hits_from_S(value_similarity, torch.LongTensor(test_pair))
        cache_dir = Path("data/%s/cache/" % i)
        cache_dir.mkdir(exist_ok=True)
        cache_data(attr_similarity, cache_dir / "attr_similarity")
        cache_data(value_similarity, cache_dir / "value_similarity")
