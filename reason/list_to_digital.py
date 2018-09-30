import jieba
import pickle

jieba.enable_parallel(4)


def test_pickle():
    with open('ID.pkl', 'rb') as f:
        ID = pickle.load(f)
    with open('train_dialogues_and_pos.pkl', 'rb') as f:
        (train_dialogues, train_dialogues_pos) = pickle.load(f)
    with open('testB_dialogues_and_pos.pkl', 'rb') as f:
        (test_dialogues, test_dialogues_pos) = pickle.load(f)
    print(ID[:10])
    print(train_dialogues[:10])
    print(train_dialogues_pos[:10])
    print(test_dialogues[:10])
    print(test_dialogues_pos[:10])
    return train_dialogues, train_dialogues_pos, test_dialogues, test_dialogues_pos


def encode_dict():
    with open('train_dialogues_and_pos.pkl', 'rb') as f:
        (train_dialogues, train_dialogues_pos) = pickle.load(f)
    with open('testB_dialogues_and_pos.pkl', 'rb') as f:
        (test_dialogues, test_dialogues_pos) = pickle.load(f)
    total_list = train_dialogues + train_dialogues_pos + test_dialogues_pos + test_dialogues
    print(len(total_list))
    if len(total_list) != 50000:
        exit('total_list != 50000')
    total_set = set()
    for a in total_list:
        for b in a:
            for c in b:
                total_set.add(c)
    print(len(total_set))
    total_dict = dict()
    for a in total_set:
        total_dict[a] = len(total_dict)
    print(len(total_dict))
    if total_dict[list(total_dict.keys())[5211]] != 5211:
        exit(-5211)
    return total_dict


total_dict = encode_dict()
train_dialogues, train_dialogues_pos, test_dialogues, test_dialogues_pos = test_pickle()
print(len(total_dict))


# print(train_dialogues[0]) one sample is a double-list
def triple_list_to_digital(train_dialogues, total_dict):
    tri_a = []
    tri_b = []
    tri_c = []
    for a in train_dialogues:
        for b in a:
            for c in b:
                c_index = total_dict[c]
                tri_c.append(c_index)
            tri_b.append(tri_c)
            tri_c = []
        tri_a.append(tri_b)
        tri_b = []
    print(len(tri_a))
    print(train_dialogues[0])
    print(tri_a[0])
    return tri_a


train_dialogues = triple_list_to_digital(train_dialogues, total_dict)
test_dialogues = triple_list_to_digital(test_dialogues, total_dict)
train_dialogues_pos = triple_list_to_digital(train_dialogues_pos, total_dict)
test_dialogues_pos = triple_list_to_digital(test_dialogues_pos, total_dict)
with open('4y.pkl', 'rb') as f:
    (encoded_Y1, y1_categorical, encoded_Y2, y2_categorical) = pickle.load(f)
with open('xxxxyy.pkl', 'wb') as f:
    pickle.dump(
        (train_dialogues, train_dialogues_pos, test_dialogues, test_dialogues_pos, y1_categorical, y2_categorical), f)
