import arff
from collections import deque
import json
import treepredict
import copy

#将多标签转换成单标签，例如，将1,1,0,1,1,0转换成54
def label_encoding(data_labels):
    new_label = ''
    for label in data_labels:
        new_label = new_label + label
    return new_label

def label_decoding(single_label):
    label_list = []
    for i in range(0, len(single_label)):
        label_list.append(single_label[i])
    return label_list



"""从文件中读取数据(arff文件)，包括训练集和测试集

"""
def read_data(filename, label_count,file_type = arff.DENSE):
    with open(filename, 'r') as data_file:
        data = arff.load(data_file, return_type=file_type)
        attributes = data['attributes'] #此为list，属性是有序的
        attributes_list = attributes[:(len(attributes) - label_count)]
        label_list = attributes[(len(attributes) - label_count):]
        train_data = data['data']
    return (attributes_list, label_list,train_data)

def translate_label_multiclass(data, label_count):
    for row in data:
        new_label = label_encoding(row[(len(row) - label_count):])
        for i in range(0, label_count): #去掉原来的label_count个标签
            row.pop()
        row.append(new_label)   #替换成单个新的标签new_label
    return data

def translate_label_binary(examples, label_count, label_index):
    result_examples = copy.deepcopy(examples)
    for row in result_examples:
        row = translate_label_binary_line(row, label_count, label_index)

    return result_examples

def translate_label_binary_line(example, label_count, label_index):
    result_example = copy.deepcopy(example)
    labels = example[(len(example) - label_count):] #获取原来的label_count个标签
    for i in range(0, label_count): #先将原标签去掉
        example.pop()
    example.append(labels[label_index]) #将指定的标签label_index重新加入
    return result_example


def tree2array(tree):
    array = []
    root = [tree.col, tree.value, tree.results, 1, 2]
    array.append(root)
    queue = deque()
    queue.append((tree.tb, 0, True))
    queue.append((tree.fb, 0, False))
    while len(queue) != 0:
        (treenode, father_index, LR_flag) = queue.popleft()
        node = [treenode.col, treenode.value, treenode.results, -1, -1]
        array.append(node)
        index = len(array) - 1

        if LR_flag: array[father_index][3] = index
        else : array[father_index][4] = index

        if treenode.results == None:    #没有结果集说明不是叶节点，也就是有子节点，将子节点放入queue中
            queue.append((treenode.tb, index, True))
            queue.append((treenode.fb, index, False))
    return array

def list2tree(tree_list, index = 0):
    tree = None
    if index < len(tree_list):  #防止错误参数
        tree = treepredict.decisionnode(col     = tree_list[index][0],
                                        value   = tree_list[index][1],
                                        results = tree_list[index][2],
                                        tb      = None,
                                        fb      = None)
#       print(tree.col, tree.value, tree.results)
        tb_index = tree_list[index][3]
        fb_index = tree_list[index][4]
        if tree.results == None and fb_index < len(tree_list):
            tree.tb = list2tree(tree_list, tb_index)
            tree.fb = list2tree(tree_list, fb_index)
    return tree

def store_tree(filename, tree_list):
    with open(filename, mode='w') as file:
        file.write(json.dumps(tree_list))

def load_tree(filename):
    with open(filename, mode='r') as file:
        str = file.read()
        tree_list = json.loads(str)
    return tree_list