# coding: utf-8
import os
import time
import random
import torch.optim
import numpy as np
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from src.train_and_evaluate import USE_CUDA, compute_prefix_tree_result
from src.tree_train_and_evaluate import train_tree, evaluate_tree
from src.tree_models import Prediction, GenerateNode, Merge
from src.EnhanceBertModel import EnhanceBertModel
from src.expressions_transfer import from_infix_to_prefix
from src.utils import time_since, read_json, load_raw_data, load_attributes
from src.pre_data import transfer_num, prepare_data, prepare_train_batch

batch_size = 32
embedding_size = 128
hidden_size = 768
n_epochs = 80
learning_rate = 5e-5
weight_decay = 1e-5
beam_size = 5
seed = 42
ori_path = './data/'
prefix = '23k_processed.json'
output_dir = 'models/seq2tree-attributes/'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_test_fold(ori_path, prefix,data, pairs, attributes):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []

    for item,pair,attr in zip(data, pairs, attributes):
        pair = list(pair)
        pair.append(attr)
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

data = load_raw_data("data/Math_23K.json")
attributes = load_attributes('data/Math_23K_attributes.json')

pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3], p[1]))

pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path, prefix, data, pairs,attributes)

pairs_tested = test_fold
pairs_trained = train_fold

tokenizer = BertTokenizer.from_pretrained('models/MWP-BERT')
new_tokens = [f"#{i}" for i in range(30)]
tokenizer.add_tokens(new_tokens)

output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, generate_nums, copy_nums, tokenizer)

set_seed(seed=seed)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize models
encoder = EnhanceBertModel.from_pretrained('models/MWP-BERT')
encoder.resize_token_embeddings(len(tokenizer))

predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_ground_paramters = [
    {'params': [p for n,p in encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n,p in encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

encoder_optimizer = torch.optim.AdamW(optimizer_ground_paramters, lr=learning_rate, eps=1e-8)
predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate*10, weight_decay=weight_decay)
generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate*10, weight_decay=weight_decay)
merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate*10, weight_decay=weight_decay)

encoder_scheduler = get_linear_schedule_with_warmup(encoder_optimizer, num_warmup_steps=0.1*n_epochs, num_training_steps=n_epochs)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    loss_total = 0

    print("epoch:", epoch + 1)
    start = time.time()

    input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
   num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, target_batches, input_pre_batches, input_pre_lengths, target_pre_batches, attribute_pos_batches = prepare_train_batch(train_pairs, batch_size)

    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], attribute_pos_batches[idx])
        loss_total += (loss / len(input_lengths))

    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()

    print("Total loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("-" * 100)
    
    if (epoch + 1) % 5 == 0 or epoch > n_epochs - 6:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], test_batch[7], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("-" * 120)
        
        encoder.save_pretrained(output_dir)
        torch.save(predict.state_dict(), output_dir+"predict.pth")
        torch.save(generate.state_dict(), output_dir+"generate.pth")
        torch.save(merge.state_dict(), output_dir+"merge.pth")
        tokenizer.save_pretrained(output_dir)
