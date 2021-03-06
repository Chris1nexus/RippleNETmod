import collections
import os
import numpy as np
import model
from utils import *

def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = DATASET_DIR[args.dataset] + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np)

'''
def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 7:1:2
    eval_ratio = 0.1
    test_ratio = 0.2
    train_ration = 1 - test_ratio
    n_ratings = rating_np.shape[0]


    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in range(n_ratings):
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = []
    valid_indices = []
    test_indices = []
    for uid, interacted_items in user_history_dict.items():
        n_interactions = len(interacted_items)
        last_idx_train = int(n_interactions*train_ration)
        last_idx_valid = last_idx_train + (n_interactions*eval_ratio)
        i = 0
        while i < last_idx_train:
            # if user i in ratings is in KG
            if rating_np[i][0] in user_history_dict:
                train_indices.append(i)
                i += 1
        while i < last_idx_valid:
            if rating_np[i][0] in user_history_dict:
                valid_indices.append(i)
                i += 1
        while i < n_interactions:
            if rating_np[i][0] in user_history_dict:
                test_indices.append(i)
                i += 1

    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[valid_indices]
    test_data = rating_np[test_indices]
    print(train_data.shape)
    print(eval_data.shape)
    print(test_data.shape)
    return train_data, eval_data, test_data, user_history_dict
'''
def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.1
    test_ratio = 0.2
    train_ratio = 1 - (eval_ratio + test_ratio)
    n_ratings = rating_np.shape[0]

    #eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    #left = set(range(n_ratings)) - set(eval_indices)
    #test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    #train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    print(rating_np.shape)
    indices = np.argsort(rating_np[:,3])
    time_ordered_ratings = rating_np[indices,...]


    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in range(time_ordered_ratings.shape[0]):
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)


    train_indices = []
    valid_indices = []
    test_indices = []
    for user_id in user_history_dict.keys():#np.unique(time_ordered_ratings[:,0]):
        indices = np.asarray(time_ordered_ratings[:,0] == user_id).nonzero()[0]

        n_user_interactions = indices.shape[0]
        last_idx_train = int(n_user_interactions*train_ratio)
        last_idx_valid = int(last_idx_train + (n_user_interactions*eval_ratio))

        train_indices.extend(indices[:last_idx_train])
        valid_indices.extend(indices[last_idx_train:last_idx_valid])
        test_indices.extend(indices[last_idx_valid:])




    train_data = time_ordered_ratings[train_indices]
    eval_data = time_ordered_ratings[valid_indices]
    test_data = time_ordered_ratings[test_indices]
    print(train_data.shape)
    print(eval_data.shape)
    print(test_data.shape)
    return train_data, eval_data, test_data, user_history_dict

def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = KG_DIR[args.dataset]+ "/kg_final"#[args.kg] + "/kg_final"
    print(kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32,delimiter='\t')
        np.save(kg_file + '.npy', kg_np)
    print(kg_np.shape)
    n_entity = max(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
