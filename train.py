import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from model import RippleNet
from sklearn.metrics import roc_auc_score
import heapq
import time
import wandb


def train(args, data_info, show_loss):

    # UIR or UIRT FORMAT REQUIRED
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]


    dataset_splits = {'train':train_data,
                        'val':eval_data,
                        'test':test_data}    
    pos_mapping, all_items = make_user_pos_items_mapping(dataset_splits)


    model = RippleNet(args, n_entity, n_relation)
    train_time = 0
    val_time = 0
    test_time = 0
    final_test_time = 0 


    
    STORAGE_DIR = 'Ripplenet'
    import os
    import wandb
    os.makedirs(STORAGE_DIR, exist_ok=True)

    if args.wandb :
        wandb.init(project=STORAGE_DIR,
                    config=vars(args),
                       entity="chris1nexus",
                       name=args.dataset)    
        


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)

            start = 0
            epoch_start_time = time.time()


            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))
                if args.wandb :
                    wandb.log({'train_loss': loss})    
            epoch_end_time = time.time()
            train_time += epoch_end_time - epoch_start_time
            # evaluation
            train_auc, train_acc = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)

            start_time = time.time()
            eval_auc, eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
            val_time += time.time() - start_time
            start_time = time.time()
            test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)
            test_time += time.time()- start_time

            USER_TEST_BATCH = 1
            #val_auc, val_ndcg, val_precision, val_recall, val_hit_ratio 
            eval_metrics = test(sess, args, model, eval_data, ripple_set, pos_mapping, all_items, 
                                K=10, batch_size=USER_TEST_BATCH, test_split_name='val')

            test_start_time = time.time() 
            #test_auc, test_ndcg, test_precision, test_recall, test_hit_ratio = 
            test_metrics = test(sess, args, model, test_data, ripple_set, pos_mapping, all_items, 
                                K=10, batch_size=USER_TEST_BATCH, test_split_name='test')            
            test_end_time = time.time()
            final_test_time += test_end_time - start_time
            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
            
            #print('val auc: %.4f  test auc: %.4f \nval ndcg: %.4f  test ndcg: %.4f \nval recall: %.4f  test recall: %.4f \nval precision: %.4f  test precision: %.4f \nval hitratio: %.4f  test hitratio: %.4f'
            #      % (val_auc, test_auc, val_ndcg, test_ndcg, val_recall, test_recall, val_precision, test_precision, val_hit_ratio, test_hit_ratio))
            
            val_logs = {'epoch': step,
                        'train_epoch_time':epoch_end_time - epoch_start_time,
                        'train_total_time':train_time,
                        'test_epoch_time': test_end_time - start_time,
                        'test_total_time':final_test_time,
                    **test_metrics,
                    **eval_metrics,
            }


            if args.wandb :
                wandb.log(val_logs)



        print('train_time %d\nval_time %d\ntest_time %d\nfinal_test_time %d ' % (train_time, val_time, test_time, final_test_time) )

def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    #print(data[start:end, 2])
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    while start < data.shape[0]:
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))








def make_user_pos_items_mapping(dataset_splits):
       
    pos_mapping = dict()
    
    for split_name, data in dataset_splits.items():                         

        if split_name not in pos_mapping:
            pos_mapping[split_name] = dict()

        user_ids = np.unique(data[:,0])
        pos_items_split = (data[:,2] > 0)
        for user_id in user_ids:            
            indices = np.where((data[:,0] == user_id)& pos_items_split)[0]
            pos_items = set(np.unique(data[indices,1]))
            pos_mapping[split_name][user_id] = pos_items 

    all_items = set(np.unique(dataset_splits['train'][:,1])).union(
                set(np.unique(dataset_splits['val'][:,1]))).union(
                set(np.unique(dataset_splits['test'][:,1])))

    return pos_mapping, all_items



def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def get_auc(item_score, user_pos_test):
    import copy
    item_score = sorted(copy.deepcopy(item_score).items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc_res = auc(ground_truth=r, prediction=posterior)
    return auc_res    
def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))



def recall_at_k_mod(r, k, r_pred_scores):
    r_preds = [ 1 if score >=0.5 else 0 for score in r_pred_scores]
    r = np.array(r)
    r_preds = np.array(r_preds)
    FN = ((r == 1) & (r_preds == 0)).sum()
    TP = r.sum()
    den = float(TP+ FN )
    return  float(TP)/den  
def precision_at_k_mod(r, k, r_pred_scores):
    r_preds = [ 1 if score >=0.5 else 0 for score in r_pred_scores]
    r = np.array(r)
    r_preds = np.array(r_preds)
    FP = ((r == 0) & (r_preds == 1)).sum()
    TP = r.sum()
    den = float(TP+ FP ) 
    return  float(TP)/   den

'''
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)
'''
def precision_at_k(r,  k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)    

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.
    




def ranklist(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in range(len(test_items)):
        item_id = test_items[i]
        item_score[item_id ] = rating[i]

    K_max = Ks
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    r_preds = []
    for item in K_max_item_score:
        if item in user_pos_test:
            r.append(1)
        else:
            r.append(0)
        r_preds.append(item_score[item])
    auc = get_auc(item_score, user_pos_test)
    return r, auc, r_preds


def test(sess, args, model, data, ripple_set, pos_mapping, all_items, K=10, batch_size=None, test_split_name='test'):
    #result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
    #          'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}



    start = 0


    auc_list = []
    ndcg_list = []    
    precision_list = []
    recall_list = []
    hit_ratio_list = []


    test_users = np.unique(data[:,0])
    from tqdm import tqdm

    progressbar = tqdm(total=test_users.shape[0])
    while start <  test_users.shape[0]:

        items = list(all_items)
        test_user_batch = test_users[start:start+batch_size] 


        def kgat_feed_dict(args, model, data, ripple_set, 
                user_batch,
                item_batch):
            feed_dict = dict()
            

            flattened_item_list = []
            for user_test_items in item_batch:
                for item in user_test_items:
                    flattened_item_list.append(item)

            feed_dict[model.items] = flattened_item_list#np.array(item_batch).flatten()
            #feed_dict[model.labels] = data[start:end, 2]

            
            for i in range(args.n_hop):
                heads = []
                relationships = []
                tails = []
                for user, test_items in zip(user_batch, item_batch):
                    n_test_items_curr_user = len(test_items)

                    # match the KG hop information of the current user to all its associated test items
                    heads.extend([ripple_set[user][i][0] for _ in range(n_test_items_curr_user)])
                    relationships.extend([ripple_set[user][i][1] for _ in range(n_test_items_curr_user)])
                    tails.extend([ripple_set[user][i][2] for _ in range(n_test_items_curr_user)] )
                    #feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for _ in range(len(n_test_items_curr_user))]
                    #feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for _ in range(len(n_test_items_curr_user))]
                    #feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for _ in range(len(n_test_items_curr_user))]   

                feed_dict[model.memories_h[i]] = heads
                feed_dict[model.memories_r[i]] = relationships
                feed_dict[model.memories_t[i]] = tails      
            #for i in range(args.n_hop):
            #    feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
            #    feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
            #    feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
            return feed_dict
        #print('d')
        #print(len(items))
        #items_mod = np.array(items).reshape(1,-1)
        #item_batch = np.repeat(items_mod, batch_size, axis=0)     
        #print(item_batch.shape)       
        #item_batch = item_batch.reshape(-1)
        #print(item_batch.shape)



        user_items_pairings = []
        item_batch = []
        for user_id in test_user_batch:
            train_pos_items = pos_mapping['train'].get(user_id, [])
            test_pos_items = pos_mapping[test_split_name].get(user_id, [])
            
            testable_items = (all_items - set(train_pos_items))
            for split_name, users in pos_mapping.items():
                if split_name not in ['train', test_split_name]:
                    testable_items = testable_items - set(users.get(user_id, []))
            test_items = list( testable_items )
            
            user_items_pairings.append((user_id, test_pos_items, test_items))
            item_batch.append(test_items)

        # test_user_batch as [batch_size,1], 
        # item_batch as [batch_size, test_items]
        # the i-th row of item_batch represents the items to be tested for the i-th user of test_user_batch             
        scores = model.inference(sess, kgat_feed_dict(args, model, data, ripple_set, test_user_batch, item_batch))
        #scores = scores.reshape(-1,len(items))
        offset = 0 

        
        
        metrics_fn_dict = {'ndcg': lambda r,K:ndcg_at_k(r, K) ,
                                'recall':lambda r,K, n_pos_items:recall_at_k(r, K, n_pos_items),
                                'precision':lambda r,K:precision_at_k(r, K),
                                'hit_ratio':lambda r,K:hit_at_k(r, K),}

        metrics_dict = dict()

        for user_id, test_pos_items, test_items in user_items_pairings:
            n_test_items = len(test_items)

            curr_item_scores = scores[offset:(offset+n_test_items)]

            for k in [10,20,30,40]:
                r, auc, r_pred_scores = ranklist(test_pos_items, test_items, curr_item_scores, k)
                for metric_name, metric_fn in metrics_fn_dict.items():
                    key = f'{metric_name}@{k}'
                    
                    if key not in metrics_dict:
                        metrics_dict[key] = []

                    if metric_name == 'recall':
                        if len(test_pos_items) > 0:
                            metrics_dict[key].append(metric_fn(r,k, len(test_pos_items)))
                    else:
                        metrics_dict[key].append(metric_fn(r,k ))

            #r, auc, r_pred_scores = ranklist(test_pos_items, test_items, curr_item_scores, K)
            #ndcg_list.append(ndcg_at_k(r, K))
            #precision_list.append(precision_at_k_mod(r, K, r_pred_scores ))#r, K))
            #precision_list.append(precision_at_k(r, K))
            #if sum(r) > 0:
            #ret_val = recall_at_k_mod(r, K, r_pred_scores )
            #if len(test_pos_items) > 0:
            #    ret_val = recall_at_k(r, K, len(test_pos_items) )
            #    recall_list.append(ret_val  )
            #hit_ratio_list.append(hit_at_k(r, K))
            #auc_list.append(auc )
            
            offset += n_test_items

        start += batch_size

        progressbar.update(len(test_user_batch)  )
        
    out_metrics = dict()
    for metric_name, metric_values in metrics_dict.items():
        out_metrics[metric_name] = np.mean(metric_values)

    progressbar.close()

    return out_metrics #float(np.mean(ndcg_list)), float(np.mean(precision_list)),float(np.mean(recall_list)),float(np.mean(hit_ratio_list))


