import re
import copy
import random
from transformers import BertTokenizer


PAD_token = 0
num_signal = [f"#{i}" for i in range(30)]


"""
    class to save the vocab and two dict: the word->index and index->word
"""
class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    # Add words of sentence to vocab
    def add_sen_to_vocab(self, sentence):
        for word in sentence:
            if re.search("#\d+|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    # Build the output lang vocab and dict
    def build_output_lang_for_tree(self, generate_num, copy_nums):
        self.num_start = len(self.index2word)
        self.index2word = self.index2word + generate_num + ["#" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


# Transfer num into "#n"
def transfer_num(data):
    print("Transfer numbers...")

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0

    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split()
        equations = d["equation"][2:]
        i = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("#{}".format(str(i)))
                i += 1
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            # elif pos and pos.start() != 0:
            #     input_seq.append(s[:pos.start()])
            #     nums.append(s[pos.start():pos.end()])
            #     input_seq.append("#{}".format(str(i)))
            #     i += 1
            #     if pos.end() < len(s):
            #         input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag(equations, nums_fraction, nums)

        # Tag the num which is generated
        for s in out_seq:
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        # (i,'j')---[(0,'xxx'),(1,'#0'),(2,'xxx')]
        for i, j in enumerate(input_seq):
            if j in num_signal:
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        pairs.append((input_seq, out_seq, nums, num_pos))

    # Not in the nums, but appears more than 5 times
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums

# Seg the equation and tag the num
def seg_and_tag(st, nums_fraction, nums):
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag(st[:p_start], nums_fraction, nums)
            if nums.count(n) == 1:
                res.append("#"+str(nums.index(n)))
            else:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag(st[p_end:], nums_fraction, nums)
            return res
        
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        if nums.count(st_num) == 1:
            res.append("#"+str(nums.index(st_num)))
        else:
            res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag(st[p_end:], nums_fraction, nums)
        return res

    for ss in st:
        res.append(ss)
    return res

# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])

    return res

# Prepare the Math23K data
def prepare_data(pairs_trained, pairs_tested, generate_nums, copy_nums, tokenizer: BertTokenizer):
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        # if pair is not None
        if pair[-1]:
            output_lang.add_sen_to_vocab(pair[1])
    
    output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    
    # pair=(input, output_prefix, nums, nums_pos, output_infix, attributes)
    for pair in pairs_trained:
        num_stack = []
                
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)
            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        input = tokenizer(pair[0], return_tensors="pt", add_special_tokens=False, is_split_into_words=True)
        input_length = input["input_ids"].squeeze().size(0)

        num_pos = []
        for idx,i in enumerate(input['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) in num_signal:
                num_pos.append(idx)
        
        num_stack.reverse()
        output_cell = indexes_from_sentence(output_lang, pair[1])

        target = " ".join(pair[4])
        target_pre = " ".join(pair[1])

        input_pre = tokenizer(['pre'] + pair[0], return_tensors="pt", add_special_tokens=False, is_split_into_words=True)
        input_pre_length = input_pre["input_ids"].squeeze().size(0)

        train_pairs.append((input, input_length, output_cell, len(output_cell),
                            pair[2], num_pos, num_stack, target, input_pre, input_pre_length, target_pre, pair[5]))
    
    print('Indexed %d words in output' % (output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []

        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        input = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
        input_length = input["input_ids"].squeeze().size(0)
        num_pos = []
        for idx,i in enumerate(input['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) in num_signal:
                num_pos.append(idx)

        num_stack.reverse()
        output_cell = indexes_from_sentence(output_lang, pair[1])

        test_pairs.append((input, input_length, output_cell, len(output_cell),
                           pair[2], num_pos, num_stack, pair[5]))
        
    print('Number of testind data %d' % (len(test_pairs)))
    return output_lang, train_pairs, test_pairs

# Prepare data for Math23K and Ape-clean fusion
def prepare_data_ape(pairs_trained, pairs_tested, pairs_tested_ape, generate_nums, copy_nums, tokenizer: BertTokenizer):
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_ape_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        # if pair is not None
        if pair[-1]:
            output_lang.add_sen_to_vocab(pair[1])
    
    output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    
    for pair in pairs_trained:
        num_stack = []
                
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)
            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        input = tokenizer(pair[0], return_tensors="pt", add_special_tokens=False, is_split_into_words=True)
        input_length = input["input_ids"].squeeze().size(0)

        num_pos = []
        for idx,i in enumerate(input['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) in num_signal:
                num_pos.append(idx)
        
        num_stack.reverse()
        output_cell = indexes_from_sentence(output_lang, pair[1])

        target = " ".join(pair[4])
        target_pre = " ".join(pair[1])

        input_pre = tokenizer(['pre'] + pair[0], return_tensors="pt", add_special_tokens=False, is_split_into_words=True)
        input_pre_length = input_pre["input_ids"].squeeze().size(0)

        train_pairs.append((input, input_length, output_cell, len(output_cell),
                            pair[2], num_pos, num_stack, target, input_pre, input_pre_length, target_pre, pair[5]))
        
    print('Indexed %d words in output' % (output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []

        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        input = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
        input_length = input["input_ids"].squeeze().size(0)

        num_pos = []
        for idx,i in enumerate(input['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) in num_signal:
                num_pos.append(idx)

        num_stack.reverse()
        output_cell = indexes_from_sentence(output_lang, pair[1])

        test_pairs.append((input, input_length, output_cell, len(output_cell),
                           pair[2], num_pos, num_stack, pair[5]))
        
    print('Number of 23k testing data %d' % (len(test_pairs)))

    for pair in pairs_tested_ape:
        num_stack = []

        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        input = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
        input_length = input["input_ids"].squeeze().size(0)

        num_pos = []
        for idx,i in enumerate(input['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) in num_signal:
                num_pos.append(idx)

        num_stack.reverse()
        output_cell = indexes_from_sentence(output_lang, pair[1])

        test_ape_pairs.append((input, input_length, output_cell, len(output_cell),
                           pair[2], num_pos, num_stack, pair[5]))
        
    print('Number of ape testing data %d' % (len(test_pairs)))
    
    return output_lang, train_pairs, test_pairs, test_ape_pairs

# Pad with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

# Prepare the batches for training
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    # Shuffle the pairs
    random.shuffle(pairs)
    
    pos = 0
    input_lengths = []
    input_pre_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    input_pre_batches = []
    output_batches = []
    # Save the num stack which
    num_stack_batches = []
    num_pos_batches = []
    attribute_pos_batches = []
    num_size_batches = []
    num_value_batches = []
    target_batches = []
    target_pre_batches = []
    

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        input_pre_length = []
        output_length = []

        for _, i, _, j, _, _, _, _, _, k, _, _ in batch:
            output_length.append(j)
            input_length.append(i)
            input_pre_length.append(k)
        output_lengths.append(output_length)
        input_lengths.append(input_length)
        input_pre_lengths.append(input_pre_length)

        output_len_max = max(output_length)

        input_batch = []
        input_pre_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        attribute_pos_batch = []
        num_size_batch = []
        num_value_batch = []
        target_batch = []
        target_pre_batch = []

        for i, _, j, lj, num, num_pos, num_stack, target, k, _, target_pre, attribute in batch:
            num_batch.append(len(num))
            input_batch.append(i)
            input_pre_batch.append(k)
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            attribute_pos_batch.append(attribute)
            num_size_batch.append(len(num_pos))
            num_value_batch.append(num)
            target_batch.append(target)
            target_pre_batch.append(target_pre)
            
        input_batches.append(input_batch)
        input_pre_batches.append(input_pre_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        attribute_pos_batches.append(attribute_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        target_batches.append(target_batch)
        target_pre_batches.append(target_pre_batch)
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, \
        target_batches, input_pre_batches, input_pre_lengths, target_pre_batches, attribute_pos_batches