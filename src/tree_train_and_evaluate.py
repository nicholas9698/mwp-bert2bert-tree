import copy
import torch
import torch.nn as nn
from src.tree_models import TreeNode
from src.EnhanceBertModel import EnhanceBertModel
from src.masked_cross_entropy import masked_cross_entropy
from src.train_and_evaluate import get_all_number_encoder_outputs, generate_tree_input, copy_list, TreeEmbedding, TreeBeam, USE_CUDA, MAX_OUTPUT_LENGTH

def create_encoder_inputs(inputs, input_lengths, num_pos, attribute_pos):
    input_length_max = max(input_lengths)
    input_ids = []
    attention_mask = []
    num_pos_ids = []
    attribute_pos_ids = []

    for idx,item in enumerate(inputs):
        input_id = item["input_ids"].squeeze()
        mask = item["attention_mask"].squeeze()
        zeros = torch.zeros(input_length_max - input_id.size(0))
        padded = torch.cat([input_id.long(), zeros.long()])
        input_ids.append(padded)
        padded = torch.cat([mask.long(), zeros.long()])
        attention_mask.append(padded)
        if num_pos != None:
            num_pos_id = torch.ones(input_id.size(0), dtype=torch.long)
            padded = torch.cat((num_pos_id.long(), zeros.long()), dim=0).tolist()
            for num in num_pos[idx]:
                padded[num] = 2
            padded = torch.LongTensor(padded)
            num_pos_ids.append(padded)
        if attribute_pos != None:
            attribute_pos_id = torch.ones(input_id.size(0), dtype=torch.long)
            padded = torch.cat((attribute_pos_id.long(), zeros.long()), dim=0).tolist()
            for attr in attribute_pos[idx]:
                padded[attr] = 2
            padded = torch.LongTensor(padded)
            attribute_pos_ids.append(padded)
        
    input_ids = torch.stack(input_ids, dim=0).long().cuda()
    attention_mask = torch.stack(attention_mask, dim=0).long().cuda()
    if num_pos != None:
        num_pos_ids = torch.stack(num_pos_ids, dim=0).long().cuda()
    if attribute_pos != None:    
        attribute_pos_ids = torch.stack(attribute_pos_ids, dim=0).long().cuda()
    
    return input_ids, attention_mask, num_pos_ids, attribute_pos_ids

def create_evaluate_inputs(input, num_pos, attribute_pos):
    input_ids = input["input_ids"].long().cuda()
    attention_mask = input["attention_mask"].long().cuda()
    if num_pos != None:
        num_pos_ids = torch.ones(input["input_ids"].squeeze().size(0), dtype=torch.long).tolist()
        for num in num_pos:
            num_pos_ids[num] = 2
        num_pos_ids = torch.LongTensor(num_pos_ids).unsqueeze(0).cuda()
    if attribute_pos != None:
        attribute_pos_ids = torch.ones(input["input_ids"].squeeze().size(0), dtype=torch.long).tolist()
        for attr in attribute_pos:
            attribute_pos_ids[attr] = 2
        attribute_pos_ids = torch.LongTensor(attribute_pos_ids).unsqueeze(0).cuda()
    
    return input_ids, attention_mask, num_pos_ids, attribute_pos_ids

def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder: EnhanceBertModel, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, attribute_pos_batch):

    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()

    input_ids, attention_mask, num_pos_ids, attribute_pos_ids = create_encoder_inputs(input_batch, input_length, num_pos, attribute_pos_batch)
    last_hidden_state = encoder(input_ids=input_ids, attention_mask=attention_mask, num_pos_ids=num_pos_ids, attribute_pos_ids=attribute_pos_ids)[0]
    encoder_outputs = last_hidden_state.transpose(0, 1)
    problem_output = encoder_outputs.mean(0)
    
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.config.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    loss.backward()

    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()

def evaluate_tree(input_batch, input_length, generate_nums, encoder: EnhanceBertModel, predict, generate, merge, output_lang, num_pos, attribute_pos, beam_size=5, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.BoolTensor(1, input_length).fill_(0)
    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    input_ids, attention_mask, num_pos_ids, attribute_pos_ids = create_evaluate_inputs(input_batch, num_pos, attribute_pos)
    encoder_outputs = encoder(input_ids, attention_mask, num_pos_ids=num_pos_ids, attribute_pos_ids=attribute_pos_ids)[0].transpose(0, 1)
    problem_output = encoder_outputs.mean(0)

    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue

            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out