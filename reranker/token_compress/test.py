import torch

# right padding
average_num = 2
hidden_states = torch.randn(3, 10, 4096)
weights = torch.randn(3, 8, 10, 10)
attention_mask = torch.tensor([[1] * 8] * 3)
attention_mask[0][-1] = 0
attention_mask[1][-1] = 0
query_lengths = torch.tensor([1, 2, 1])
prompt_lengths = torch.tensor([1, 1, 3])

passage_lengths = torch.tensor([int(sum(attention_mask[batch])) - query_lengths[batch] -
                   prompt_lengths[batch] for batch in range(hidden_states.shape[0])])
retain_lengths = torch.tensor([(passage_num + average_num - 1) // average_num for passage_num in passage_lengths])
max_length = max([query_lengths[batch] + prompt_lengths[batch] +
                                  retain_lengths[batch] for batch in range(hidden_states.shape[0])])
max_passage_length = max(passage_lengths)
padding_lengths = torch.tensor([max_length - query_lengths[batch] - prompt_lengths[batch] -
                       retain_lengths[batch] for batch in range(hidden_states.shape[0])])
padding_nums = torch.tensor([hidden_states.shape[1] - int(sum(attention_mask[batch])) for batch in range(hidden_states.shape[0])])
start_locs = query_lengths
end_locs = torch.tensor([hidden_states.shape[1] - prompt_lengths[batch] - padding_nums[batch] for batch in range(hidden_states.shape[0])])

weights = weights[:, :, -1, :]
# print(weights)
# weights = torch.sum(weights, dim=1)
# print(weights)
# print(weights.shape)

new_hidden_states = torch.zeros((hidden_states.shape[0], max_length, hidden_states.shape[-1])).to(hidden_states.device)
new_attention_mask = torch.ones((hidden_states.shape[0], max_length)).to(attention_mask.device)
# for i in range(hidden_states.shape[0]):
#     new_hidden_states[i, :query_lengths[i]] = hidden_states[i, :query_lengths[i]]
#     new_hidden_states[i, query_lengths[i] + retain_lengths[i]: query_lengths[i] + retain_lengths[i] + prompt_lengths[i]
#             ] = hidden_states[i, end_locs[i]: end_locs[i] + prompt_lengths[i]]
#     new_attention_mask[i, -padding_nums[i]:] = attention_mask[i, -padding_nums[i]:]

query_lengths = torch.tensor(query_lengths)
prompt_lengths = torch.tensor(prompt_lengths)
retain_lengths = torch.tensor(retain_lengths)
padding_nums = torch.tensor(padding_nums)
# Calculate the indices for slicing
start_indices = torch.arange(hidden_states.shape[0]).unsqueeze(1).long()  # 转换为整数类型
end_indices = (query_lengths + retain_lengths).unsqueeze(1) + prompt_lengths.unsqueeze(1)

# Update new_hidden_states
new_hidden_states[start_indices, :query_lengths.unsqueeze(1).long()] = hidden_states[start_indices, :query_lengths.unsqueeze(1).long()]
new_hidden_states[start_indices, query_lengths.unsqueeze(1).long() + retain_lengths.unsqueeze(1): end_indices] = hidden_states[start_indices, end_locs.unsqueeze(1): end_locs.unsqueeze(1) + prompt_lengths.unsqueeze(1)]

# Update new_attention_mask
new_attention_mask[start_indices, -padding_nums.unsqueeze(1):] = attention_mask[start_indices, -padding_nums.unsqueeze(1):]

hidden_states_for_passage = torch.zeros((hidden_states.shape[0], max_passage_length, hidden_states.shape[-1])).to(hidden_states.device)
weights_for_passage = torch.zeros((weights.shape[0], weights.shape[1], (max_passage_length + average_num - 1) // average_num * average_num)).to(weights.device)
for i in range(hidden_states.shape[0]):
    hidden_states_for_passage[i, :passage_lengths[i]] = hidden_states[i, query_lengths[i]: query_lengths[i] + passage_lengths[i]]
    weights_for_passage[i, :, :passage_lengths[i]] = weights[i, :, query_lengths[i]: query_lengths[i] + passage_lengths[i]]
weights_for_passage_type1 = weights_for_passage.view()
print(hidden_states_for_passage)
print(passage_lengths)


print(weights_for_passage)

# print(new_attention_mask)
# print(new_hidden_states)