import argparse
import os

import numpy as np

RATING_FILE_NAME = dict({'ml1m': 'ratings.dat', 
    'ml1m_freebase':'ratings.dat',
    'yago':'ratings.dat',
    'lfm1m': 'ratings.txt',
    'book': 'BX-Book-Ratings.csv', 
    'news': 'ratings.txt'})
SEP = dict({'ml1m': '::', 'book': ';', 'news': '\t',
    'yago': '::',
    'lfm1m': '\t',
    'ml1m_freebase':'::'
    })
THRESHOLD = dict({'ml1m': 0, 'book': 0, 'news': 0,'yago': 0,
                'lfm1m':0, 'ml1m_freebase':0})


entity_id2index = dict()
relation_id2index = dict()
item_index_old2new = dict()
def read_item_index_to_entity_id_file(dataset, raw_kg_path):

    file = os.path.join(raw_kg_path, "item_index2entity_id_rehashed.txt")
    #file = '../data/' + DATASET + '/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        kg_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[kg_id] = i
        i += 1


def convert_rating(dataset, raw_kg_path):

    file = os.path.join(raw_kg_path, RATING_FILE_NAME[dataset])

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split(SEP[dataset])

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        timestamp = float(array[3])
        if rating >= THRESHOLD[dataset]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = dict()#set()
            if item_index not in user_pos_ratings[user_index_old]:
                user_pos_ratings[user_index_old][item_index] = list()
            user_pos_ratings[user_index_old][item_index].append(timestamp)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = dict()#set()
            if item_index not in user_neg_ratings[user_index_old]:
                user_neg_ratings[user_index_old][item_index] = list()                
            user_neg_ratings[user_index_old][item_index].append(timestamp)

    print('converting rating file ...')
    print(os.path.join(raw_kg_path, 'ratings_final.txt'))
    writer = open(os.path.join(raw_kg_path, 'ratings_final.txt'), 'w+', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_dict in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]


        used_timestamps = []
        for item, timestamps in pos_item_dict.items():
            latest_tstamp = max(timestamps)
            writer.write('%d\t%d\t1\t%d\n' % (user_index, item, latest_tstamp ))
            used_timestamps.append(latest_tstamp)
        unwatched_set = item_set - set(pos_item_dict.keys())


        if user_index_old in user_neg_ratings:
            #for k in user_neg_ratings[user_index_old].keys():
            #    del unwatched_dict[k]
            unwatched_set = unwatched_set - set(user_neg_ratings[user_index_old].keys())
        
        unwatched_items = np.random.choice(list(unwatched_set), size=min(len(pos_item_dict), len(unwatched_set)), replace=False)
        # associate to all positive items a negative one with the same timestamp, in order to have a balanced negative sampling for the creation 
        # of the training, val, test sets 
        # these are FAKE timestamps in the sense that an unwatched item cannot have a proper timestamp, but it is necessary for time-ordering while maintaining
        # such positive to negative item balance in all sets
        for item, timestamp in zip(unwatched_items,used_timestamps):
            writer.write('%d\t%d\t0\t%d\n' % (user_index, item, timestamp ))
        #for item in np.random.choice(list(unwatched_set), size=min(len(pos_item_set), len(unwatched_set)), replace=False):
        #    writer.write('%d\t%d\t0\t%d\n' % (user_index, item, max(timestamps) ))
                        
            #writer.write('%d\t%d\t0\n' % (user_index, item))        
        '''
        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=min(len(pos_item_set), len(unwatched_set)), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
        '''
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg(raw_kg_path):
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open(os.path.join(raw_kg_path, 'kg_final.txt'), 'w+', encoding='utf-8')

    files = []
    files.append(open(os.path.join(raw_kg_path, 'kg.txt'), encoding='utf-8'))
    #files.append(open(os.path.join(raw_kg_path, 'kg_part1_rehashed.txt'), encoding='utf-8'))
    #files.append(open(os.path.join(raw_kg_path, 'kg_part2_rehashed.txt'), encoding='utf-8'))
    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]
            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(2022)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1m', help='which dataset to preprocess')
    parser.add_argument('--kg', type=str, default='dbpedia', help='which dataset to preprocess')
    args = parser.parse_args()

    dataset = args.dataset
    kg = args.kg
    raw_kg_path = f'./data/{dataset}'
    read_item_index_to_entity_id_file(dataset, raw_kg_path)
    convert_rating(dataset,  raw_kg_path)
    convert_kg(raw_kg_path)
    print('done')
