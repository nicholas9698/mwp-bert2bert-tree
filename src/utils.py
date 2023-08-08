import re
import copy
import random
import math
import json
import torch
from typing import Dict
from transformers import BertTokenizer


'''
    Load datasets
'''
def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def load_raw_data(path):
    print("Reading lines...")
    f = open(path, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            data.append(data_d)
            js = ""
    return data

def load_ape_data(path):
    print("Reading lines...")
    f = open(path, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 8 == 0:  # every 8 line is a json
            data_d = json.loads(js)
            data.append(data_d)
            js = ""
    return data

# load numerical corroletion attributes
def load_attributes(path):
    f = open(path, encoding="utf-8")

    data = []
    for s in f:
        data_d = json.loads(s)
        data.append(data_d)
    
    attributes = []
    for d in data:
        temp = []
        for l in d["attributes_pos"]:
            temp.extend(l)
        attributes.append(temp)
    return attributes

# compute time
def time_since(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

# translate unicode to utf8 encoding, show datatime
def unicode2utf8(file_path = 'data/math23k/test23k_processed.json', file_out_path = 'data/math23k/test23k_processed_utf8.json'):   
    with open(file_path, 'r') as f:
        whole_data = json.load(f)

    with open(file_out_path, 'w' ,encoding='utf-8') as f:
        for d in whole_data:
            json.dump(d,f,indent=4,ensure_ascii=False)
            f.write('\n')


'''
    Seq2seq compute expression
'''
# fill origin nums to expression where '#i' exist
def expression_fill_nums(expression: str, num_dict: Dict):
    exp = expression.strip().split(' ')
    result = ""
    for item in exp:
        if item in num_dict:
            result += (str(num_dict[item]) + ' ')
        elif item == '^':
            result += ('**' + ' ')
        else:
            result += (item + ' ')

    return result[:-1]

def compare_result(res: str, tar: str, num_dict: Dict):
    res = res.strip().lower()
    tar = tar.strip().lower()
    res = re.sub("[\]\}]", ")", re.sub("[\[\{]", "(", res))
    tar = re.sub("[\]\}]", ")", re.sub("[\[\{]", "(", tar))
    if res == tar:
        return True, True
    else:
        res = expression_fill_nums(res, num_dict)
        tar = expression_fill_nums(tar, num_dict)
        try:
            if abs(eval(tar) - eval(res)) < 1e-4:
                return False, True
        except:
            return False, False
    
    return False, False


'''
    Data reading for bert2bert and bert2bert-tree
'''            
def read_data_and_target(file_path: str, train=True, origin_path=None):
    data_and_targets = []
    num_dicts = []
    if train:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                temp = line.strip().split('\t')
                data_and_targets.append((temp[0], temp[1]))
    
        return data_and_targets
    
    else:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                temp = line.strip().split('\t')
                data_and_targets.append((temp[0], temp[1]))
        with open(origin_path, 'r') as f:
            data = json.load(f)
        for item in data:
            num_dict = {}
            for idx,num in enumerate(item["num_list"]):
                num_dict["#{}".format(idx)] = num
            num_dict["pi"] = 3.14
            num_dicts.append(num_dict)
        return data_and_targets, num_dicts

def read_data_and_target_attributes(file_path: str, attributes_path: str, train=True, infix=True, origin_path=None):
    data_and_targets = []
    num_dicts = []
    nums_pos = []
    attributes = []
    len_words = []
    finall_data = []
    if train:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                temp = line.strip().split('\t')
                data_and_targets.append((temp[0].strip(), temp[1].strip()))
        with open(attributes_path, 'r') as f:
            if infix:
                for line in f.readlines():
                    temp = line.strip().split(',')
                    if len(temp[0])>0:
                        nums_pos.append(list(map(int, temp[0].strip().split(' '))))
                    else:
                        nums_pos.append([])
                    if len(temp[1])>0:
                        attributes.append(list(map(int, temp[1].strip().split(' '))))
                    else:
                        attributes.append([])
                    len_words.append(int(temp[2]))
            else:
                for line in f.readlines():
                    temp = line.strip().split(',')
                    if len(temp[0])>0:
                        nums_pos.append([pos+1 for pos in list(map(int, temp[0].strip().split(' ')))])
                    else:
                        nums_pos.append([])
                    if len(temp[1])>0:
                        attributes.append([pos+1 for pos in list(map(int, temp[1].strip().split(' ')))])
                    else:
                        attributes.append([])
                    len_words.append(int(temp[2])+1)
        for idx,item in enumerate(data_and_targets):
            finall_data.append((item[0], item[1], nums_pos[idx], attributes[idx], len_words[idx]))
    
        return finall_data
    
    else:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                temp = line.strip().split('\t')
                data_and_targets.append((temp[0].strip(), temp[1].strip()))
        with open(attributes_path, 'r') as f:
            for line in f.readlines():
                temp = line.strip().split(',')
                if len(temp[0])>0:
                    nums_pos.append(list(map(int, temp[0].strip().split(' '))))
                else:
                    nums_pos.append([])
                if len(temp[1])>0:
                    attributes.append(list(map(int, temp[1].strip().split(' '))))
                else:
                    attributes.append([])
                len_words.append(int(temp[2]))
        for idx,item in enumerate(data_and_targets):
            finall_data.append((item[0], item[1], nums_pos[idx], attributes[idx], len_words[idx]))
        with open(origin_path, 'r') as f:
            data = json.load(f)
        for item in data:
            num_dict = {}
            for idx,num in enumerate(item["num_list"]):
                num_dict["#{}".format(idx)] = num
            num_dict["pi"] = 3.14
            num_dicts.append(num_dict)

        return finall_data, num_dicts

def pad_sentence(data_ls, target_ls, tokenizer: BertTokenizer, USE_CUDA: bool):
    data_ls = tokenizer(data_ls, return_tensors="pt", add_special_tokens=False, padding=True)
    target_ls = tokenizer(target_ls, return_tensors="pt", add_special_tokens=False, padding=True)
    if USE_CUDA:
        inputs = {"input_ids": data_ls.input_ids.cuda(), "attention_mask": data_ls.attention_mask.cuda(), "labels": target_ls.input_ids.cuda()}
    else:
        inputs = {"input_ids": data_ls.input_ids, "attention_mask": data_ls.attention_mask, "labels": target_ls.input_ids}
    
    return inputs

def pad_sentence_attributes(data_ls, target_ls, num_pos_ls, attributes_ls, len_words_ls, tokenizer: BertTokenizer, USE_CUDA: bool):
    data_ls = tokenizer(data_ls, return_tensors="pt", add_special_tokens=False, padding=True)
    target_ls = tokenizer(target_ls, return_tensors="pt", add_special_tokens=False, padding=True)
    max_len = max(len_words_ls) 
    
    for idx in range(len(len_words_ls)):
        # if data_ls.input_ids[idx][0] == 8671:
        #     temp = torch.ones(len_words_ls[idx], dtype=torch.long)
        #     zeros = torch.zeros(max_len-len_words_ls[idx]-1, dtype=torch.long)
        #     temp = torch.cat((temp, zeros), dim=0).tolist()
        #     for num in num_pos_ls[idx]:
        #         temp[num] = 2
        #     temp.insert(0,2)
        #     num_pos_ls[idx] = temp

        #     temp = torch.ones(len_words_ls[idx], dtype=torch.long)
        #     zeros = torch.zeros(max_len-len_words_ls[idx]-1, dtype=torch.long)
        #     temp = torch.cat((temp, zeros), dim=0).tolist()
        #     for attr in attributes_ls[idx]:
        #         temp[attr] = 2
        #     temp.insert(0,2)
        #     attributes_ls[idx] = temp
        # else:
        temp = torch.ones(len_words_ls[idx], dtype=torch.long)
        zeros = torch.zeros(max_len-len_words_ls[idx], dtype=torch.long)
        temp = torch.cat((temp, zeros), dim=0).tolist()
        for num in num_pos_ls[idx]:
            temp[num] = 2
        num_pos_ls[idx] = temp

        temp = torch.ones(len_words_ls[idx], dtype=torch.long)
        zeros = torch.zeros(max_len-len_words_ls[idx], dtype=torch.long)
        temp = torch.cat((temp, zeros), dim=0).tolist()
        for attr in attributes_ls[idx]:
            temp[attr] = 2
        attributes_ls[idx] = temp

    nums_pos = torch.LongTensor(num_pos_ls)
    attributes_pos = torch.LongTensor(attributes_ls)
    
    if USE_CUDA:
        inputs = {"input_ids": data_ls.input_ids.cuda(), "attention_mask": data_ls.attention_mask.cuda(), "num_pos_ids": nums_pos.cuda(), "attribute_pos_ids": attributes_pos.cuda(), "labels": target_ls.input_ids.cuda()}
    else:
        inputs = {"input_ids": data_ls.input_ids, "attention_mask": data_ls.attention_mask, "num_pos_ids": nums_pos, "attribute_pos_ids": attributes_pos, "labels": target_ls.input_ids}
    
    return inputs

def prepare_train_batch(data, batch_size: int):
    train_pair = copy.deepcopy(data)
    random.shuffle(train_pair)
    batches = []
    pos = 0

    while pos + batch_size < len(train_pair):
        batches.append(train_pair[pos:pos+batch_size])
        pos += batch_size
    batches.append(train_pair[pos:])

    train_datas = []
    train_targets = []
    for batch in batches:
        train_data = []
        train_target = []
        for item in batch:
            train_data.append(item[0])
            train_target.append(item[1])
        train_datas.append(train_data)
        train_targets.append(train_target)

    return train_datas, train_targets

def prepare_train_batch_attributes(data, batch_size: int):
    train_pair = copy.deepcopy(data)
    random.shuffle(train_pair)
    batches = []
    pos = 0

    while pos + batch_size < len(train_pair):
        batches.append(train_pair[pos:pos+batch_size])
        pos += batch_size
    batches.append(train_pair[pos:])

    train_datas = []
    train_targets = []
    train_nums_pos = []
    train_attributes_pos = []
    len_words = []
    for batch in batches:
        train_data = []
        train_target = []
        train_num = []
        train_attributes = []
        train_len = []
        for item in batch:
            train_data.append(item[0])
            train_target.append(item[1])
            train_num.append(item[2])
            train_attributes.append(item[3])
            train_len.append(item[4])

        train_datas.append(train_data)
        train_targets.append(train_target)
        train_nums_pos.append(train_num)
        train_attributes_pos.append(train_attributes)
        len_words.append(train_len)

    return train_datas, train_targets, train_nums_pos, train_attributes_pos, len_words

def prepare_vail_test(datas, num_dicts, tokenizer: BertTokenizer, test_batch_size=8):
    data_batches = []
    target_batches = []
    num_dict_batches = []
    encode_batches = []

    pos = 0
    while pos + test_batch_size < len(datas):
        data_batches.append(datas[pos:pos+test_batch_size])
        num_dict_batches.append(num_dicts[pos:pos+test_batch_size])
        pos += test_batch_size
    data_batches.append(datas[pos:])
    num_dict_batches.append(num_dicts[pos:])

    for batch in data_batches:
        data_ls = []
        target_ls = []
        for item in batch:
            data_ls.append(item[0])
            target_ls.append(item[1])
        
        target_batches.append(target_ls)
        data_ls = tokenizer(data_ls, return_tensors="pt", add_special_tokens=False, padding=True)
        encode_batches.append(data_ls)

    return target_batches, num_dict_batches, encode_batches

def prepare_vail_test_attributes(datas, num_dicts, tokenizer: BertTokenizer, test_batch_size=8):
    data_batches = []
    target_batches = []
    num_dict_batches = []
    encode_batches = []
    nums_pos = []
    attributes = []

    pos = 0
    while pos + test_batch_size < len(datas):
        data_batches.append(datas[pos:pos+test_batch_size])
        num_dict_batches.append(num_dicts[pos:pos+test_batch_size])
        pos += test_batch_size
    data_batches.append(datas[pos:])
    num_dict_batches.append(num_dicts[pos:])

    for batch in data_batches:
        data_ls = []
        target_ls = []
        num_pos_ls = []
        attribute_ls = []
        len_word_ls = []
        for item in batch:
            data_ls.append(item[0])
            target_ls.append(item[1])
            num_pos_ls.append(item[2])
            attribute_ls.append(item[3])
            len_word_ls.append(item[4])

        target_batches.append(target_ls)
        data_ls = tokenizer(data_ls, return_tensors="pt", add_special_tokens=False, padding=True)
        encode_batches.append(data_ls)

        max_len = max(len_word_ls)
        for idx in range(len(len_word_ls)):
            temp = torch.ones(len_word_ls[idx], dtype=torch.long)
            zeros = torch.zeros(max_len-len_word_ls[idx], dtype=torch.long)
            temp = torch.cat((temp, zeros), dim=0).tolist()
            for num in num_pos_ls[idx]:
                temp[num] = 2
            num_pos_ls[idx] = temp

            temp = torch.ones(len_word_ls[idx], dtype=torch.long)
            zeros = torch.zeros(max_len-len_word_ls[idx], dtype=torch.long)
            temp = torch.cat((temp, zeros), dim=0).tolist()
            for attr in attribute_ls[idx]:
                temp[attr] = 2
            attribute_ls[idx] = temp
        nums_pos.append(torch.LongTensor(num_pos_ls))
        attributes.append(torch.LongTensor(attribute_ls))

    return target_batches, num_dict_batches, encode_batches, nums_pos, attributes


'''
    Seqseq2 expression format conversion
'''
OP_LIST = ["+", "-", "*", "/", "^"]
ORDER_DICT = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}

def from_infix_to_postfix(expression):
    if isinstance(expression, str):
        expression = expression.split(' ')
        expression = [e for e in expression if e]
    st = list()
    res = list()
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in ORDER_DICT:
            while len(st) > 0 and st[-1] not in ["(", "["] and ORDER_DICT[e] <= ORDER_DICT[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return " ".join(res)

def from_infix_to_prefix(expression):
    if isinstance(expression, str):
        expression = expression.split(' ')
        expression = [e for e in expression if e]
    st = list()
    res = list()
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in ORDER_DICT:
            while len(st) > 0 and st[-1] not in [")", "]"] and ORDER_DICT[e] < ORDER_DICT[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    
    return " ".join(res)

def from_postfix_to_infix(postfix):
    if isinstance(postfix, str):
        postfix = postfix.split(' ')
    stack = []
    for elem in postfix:
        if elem in OP_LIST:
            a, od_a = stack.pop()
            b, od_b = stack.pop()
            od_c = ORDER_DICT[elem]
            if od_a <= od_c:
                a = "( " + a + " )"
            if od_b < od_c:
                b = "( " + b + " )"
            tmp = b + " " + elem + " " + a
            stack.append((tmp, od_c))
        else:
            stack.append((elem, 3))
    assert len(stack) == 1
    return stack[-1][0]
