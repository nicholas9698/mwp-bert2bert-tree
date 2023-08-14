import os
import time
import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from src.utils import read_data_and_target_attributes, prepare_train_batch_attributes, prepare_vail_test_attributes, pad_sentence_attributes, compare_result, time_since
from src.models import BertEncoderDecoderModel

USE_CUDA = torch.cuda.is_available()

batch_size = 32
test_batch_size = 32
learning_rate = 5e-5
weight_decay = 1e-5
n_epoch = 80
seed = 42
output_dir = 'models/b2b/'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed=seed)

model = BertEncoderDecoderModel.from_encoder_decoder_pretrained('/root/autodl-tmp/MWP-BERT', '/root/autodl-tmp/MWP-BERT', tie_encoder_decoder=True)
tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/MWP-BERT')
new_tokens = [f"#{i}" for i in range(30)]
tokenizer.add_tokens(new_tokens)
model.encoder.resize_token_embeddings(len(tokenizer))
model.decoder.resize_token_embeddings(len(tokenizer))

# set model's config
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.num_beams = 5
model.config.max_new_tokens = 128
model.config.early_stopping = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_train = read_data_and_target_attributes('data/math23k/infix_math23k_processed.train', 'data/math23k/math23k_processed_attributes.train')
data_train_pre = read_data_and_target_attributes('data/math23k/pre_math23k_processed.train', 'data/math23k/math23k_processed_attributes.train', infix=False)
data_train.extend(data_train_pre)
data_test, test_num_dicts = read_data_and_target_attributes('data/math23k/infix_math23k_processed.test', 'data/math23k/math23k_processed_attributes.test', train=False, origin_path='data/test23k_processed.json')

test_target_batches, test_numdict_batches, test_encode_batches, test_nums_pos_batches, test_attributes_pos_batches = prepare_vail_test_attributes(data_test, test_num_dicts, tokenizer, test_batch_size=test_batch_size) 

if USE_CUDA:
    model.cuda()

# model size
# size = 0    
# for n, p in model.named_parameters():
#     size += p.nelement()
# print('Total parameters: {}'.format(size))

no_decay = ['bias', 'LayerNorm.weight']
optimizer_ground_paramters = [
    {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_ground_paramters, lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*n_epoch, num_training_steps=n_epoch)

model.zero_grad()

for epoch in range(n_epoch):
    loss_total = 0
    
    print('epoch', epoch+1)
    start_time = time.time()
    
    data_batches, target_batches, nums_pos_batches, attributes_pos_batches, len_words_batches = prepare_train_batch_attributes(data_train, batch_size)

    model.train()
    for idx in range(len(data_batches)):
        inputs = pad_sentence_attributes(data_batches[idx], target_batches[idx], nums_pos_batches[idx], attributes_pos_batches[idx], len_words_batches[idx], tokenizer, USE_CUDA)
        loss = model(**inputs).loss
        loss.backward()
        loss_total += (loss.item() / len(data_batches))

        optimizer.step()
        model.zero_grad()

    print("loss:", loss_total)
    print("training time", time_since(time.time() - start_time))
    print("-" * 100)
    scheduler.step()

    if (epoch+1)%5 == 0 or n_epoch-epoch<6:
        start_time = time.time()
        model.eval()
        total = 0
        exp_ac = 0
        value_ac = 0

        for idx,target_batch in enumerate(test_target_batches):
            if USE_CUDA:
                test_output = model.generate(test_encode_batches[idx].input_ids.cuda(), attention_mask=test_encode_batches[idx].attention_mask.cuda(), num_pos_ids=test_nums_pos_batches[idx].cuda(), attribute_pos_ids=test_attributes_pos_batches[idx].cuda(), max_new_tokens=128, num_beams=5, num_return_sequences=1)
            else:
                test_output = model.generate(test_encode_batches[idx].input_ids, attention_mask=test_encode_batches[idx].attention_mask, num_pos_ids=test_nums_pos_batches[idx], attribute_pos_ids=test_attributes_pos_batches[idx], max_new_tokens=128, num_beams=5, num_return_sequences=1)
            
            test_output = tokenizer.batch_decode(test_output, skip_special_tokens=True)
            
            for i,target in enumerate(target_batch):
                exp_bool, value_bool = compare_result(test_output[i], target, test_numdict_batches[idx][i])
                if exp_bool:
                    exp_ac += 1
                    value_ac += 1
                elif value_bool:
                    value_ac += 1
                total += 1

        print(exp_ac, value_ac, total)
        print("test_answer_acc", float(exp_ac) / total, float(value_ac) / total)
        print("testing time", time_since(time.time() - start_time))
        print("-" * 120)
        
        torch.save(model, output_dir+"model.pth")
        tokenizer.save_pretrained(output_dir)