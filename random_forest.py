""" 随机森林
有放回的取N条数据作为样本（原数据共N条），产生3份样本，
对于每份样本，取2*sqrt(M)个属性作为决策树的分裂属性（数据中共有M个属性），形成M / (2 * sqrt(M))棵CART树

"""
import treepredict
import preprocessor
import postprocessor
import arff
import random
import math
import copy
import json
import string


train_data_file = '.\\scene\\scene-train.arff'
test_data_file = '.\\scene\\scene-test-tiny.arff'
label_count = 6
sample_copy_count = 3
TREE = 'tree'
ATTRIBUTES_INDEX = 'attributes_index'


def generate_random_sample(data):
    data_len = len(data)
    sample_data = []
    for i in range(0, data_len):
        sample_data.append(train_data[random.randint(0, data_len - 1)])
    return sample_data


""" 产生每棵树使用的属性列表
    针对属性的序号来产生
Args:
   attribute_count: 样本中属性的总个，例如11
   attribute_count_per_tree:    每棵决策树所需要的属性个数，例如3

Return:
    attributes_list_per_tree:   每棵树的属性list,例如
                                [[10,   2,  4],
                                 [3,    7,  6],
                                 [5,    8,  0],
                                 [9,    1 ]]
"""
def choose_attributes_lists(attribute_count, attribute_count_per_tree):
    attributes_list_per_tree = []
    attributes_left = list(range(0, attribute_count))
    tree_count = math.ceil(attribute_count / attribute_count_per_tree)
    for tree_index in range(0, tree_count):
        selected_attributes_list = []

        #若剩余的属性数小于attribute_count_per_tree，那么选中剩下的所有属性
        if len(attributes_left) < attribute_count_per_tree:
            selected_attributes_list = attributes_left
        else:  # 若剩余的属性数较多（大于每棵树所需属性数），则无放回的取attribute_count_per_tree个属性
            for i in range(0, attribute_count_per_tree):
                random_attribute = random.choice(attributes_left)
                attributes_left.remove(random_attribute)  # 把选中的属性从剩余属性中去除
                selected_attributes_list.append(random_attribute)   #将随机产生的属性加入选中属性list

        # 将一棵树的selected_attributes_list个属性list加入到每棵树所需属性列表(attributes_list_per_tree)中
        attributes_list_per_tree.append(selected_attributes_list)

    return attributes_list_per_tree

"""将样本集根据attribute_list，挑出部分属性，用属性序号表示

"""
def organize_sample_with_selected_attributes(data, attribute_list):
    reduced_data = []
    for index in range(0, len(data)):
        original_sample = data[index]
        one_sample = []
        for i in attribute_list:
            one_sample.append(original_sample[i])

        #将标签添加进去
        one_sample.append(original_sample[len(original_sample) - 1])
        reduced_data.append(one_sample)

    return reduced_data

def train_random_trees(train_data, origin_attribute_list, label_list, sample_copy_count, attribute_count_per_tree):
    trees = []  #用随机选取的多个训练集
    for sample_copies_index in range(0, sample_copy_count):
        sample_copy = generate_random_sample(train_data)
        #每棵决策树使用的属性集（随机）
        random_attributes_lists = choose_attributes_lists(len(origin_attribute_list), attribute_count_per_tree)
        #用不同属性集训练的决策树
        for attributes_lists_per_tree in random_attributes_lists:    #根据随机选定的属性集训练每棵决策树
            #根据当前决策树使用的属性集，重新生成训练集（只剩下用到的属性）
            reduced_data = organize_sample_with_selected_attributes(sample_copy, attributes_lists_per_tree)
            #将属性序号(attributes_lists_per_tree)转换成属性取值信息(real_attribute_list)
            real_attribute_list = []
            for index in attributes_lists_per_tree:
                real_attribute_list.append(origin_attribute_list[index])
            tree = treepredict.buildtree(reduced_data, real_attribute_list, label_list)
            tree_with_attribute_index = {'tree':tree, 'attributes_index':attributes_lists_per_tree}
            trees.append(tree_with_attribute_index)
    return trees

def classify_with_several_trees(data, trees, original_attribute_list):
    predicted_results_all = {}
    for tree in trees:
        decision_tree = tree['tree']
        attributes_index = tree['attributes_index']
        re_organized_data = []  #根据当前决策树，重新组织数据(只留下这棵决策树用到的属性和标签)
        re_organized_attribute_value_list = []
        for index in attributes_index:
            re_organized_data.append(data[index])
            re_organized_attribute_value_list.append(original_attribute_list[index])
        re_organized_data.append(data[len(data) - 1])   #将标签加入重新组织的数据中
        predicted_results = treepredict.classify(re_organized_data, decision_tree, re_organized_attribute_value_list)
        #将当前这棵树的预测结果合并到总的预测结果中去
        for result in predicted_results.keys():
            predicted_results_all[result] = predicted_results_all.get(result, 0) + predicted_results[result]

    print('predicted_results_all:', predicted_results_all)
    single_label_result = treepredict.post_classify(predicted_results_all)
    return single_label_result

def store_random_trees(random_trees, filename_prefix):
    for i in range(0, len(random_trees)):
        decision_tree = random_trees[i][TREE]
        attributes_index = random_trees[i][ATTRIBUTES_INDEX]
        decision_tree_in_list = preprocessor.tree2array(decision_tree)
        decision_tree_filename = filename_prefix + TREE + str(i)
        attributes_index_filename = filename_prefix + ATTRIBUTES_INDEX + str(i)
        attributes_index_in_list = json.dumps(attributes_index)
        preprocessor.store_tree(decision_tree_filename, decision_tree_in_list)
        with open(attributes_index_filename, mode = 'w') as attributes_index_file:
            attributes_index_file.write(attributes_index_in_list)



def load_random_trees(filename_prefix, tree_count):
    random_trees = []
    for index in range(0, tree_count):
        decision_tree_filename = filename_prefix + TREE + str(index)
        attributes_index_filename = filename_prefix + ATTRIBUTES_INDEX + str(index)
        decision_tree_in_list = preprocessor.load_tree(decision_tree_filename)
        decision_tree = preprocessor.list2tree(decision_tree_in_list)
        with open(attributes_index_filename, mode = 'r') as attributes_index_file:
            attributes_index = json.loads(attributes_index_file.read())
        random_trees.append({TREE:decision_tree, ATTRIBUTES_INDEX:attributes_index})

    return random_trees



(origin_attribute_list, label_list, train_data) = preprocessor.read_data(train_data_file, label_count, arff.DENSE)
attribute_count = len(origin_attribute_list)
attribute_count_per_tree = math.floor(math.sqrt(attribute_count) * 2)
tree_count_per_sample_copy = math.ceil(attribute_count / attribute_count_per_tree)

train_data = preprocessor.translate_label_multiclass(train_data, label_count)   #转换成单标签数据集

random_trees = train_random_trees(train_data,origin_attribute_list, label_list, sample_copy_count, attribute_count_per_tree)


forest_count = len(random_trees)
store_random_trees(random_trees, '.\\my_forest\\my_random_forest_')
loaded_random_trees = load_random_trees('.\\my_forest\\my_random_forest_', 27)


#对数据进行预测
(original_attribute_value_list, label_value_list, test_data) = preprocessor.read_data(test_data_file, label_count, arff.DENSE)
single_label_test_data = preprocessor.translate_label_multiclass(test_data, label_count)    #转换成单标签多类分类问题进行
for row in single_label_test_data:
    print('test_data row : ', row)
    predicted_label = classify_with_several_trees(row, loaded_random_trees, original_attribute_value_list)
    print(predicted_label)






