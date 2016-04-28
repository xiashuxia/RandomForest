import treepredict
import preprocessor
import postprocessor
import arff
import copy

label_count = 6
train_data_file = '.\\scene\\scene-train-tiny.arff'
test_data_file = '.\\scene\\scene-test-tiny.arff'
method = input('1 单标签；2 多个二类分类')
if method == '1':
    #读取训练集，建树(多标签转换成单标签)
    (attributes_list, label_value_list,train_data) = preprocessor.read_data(train_data_file, label_count, arff.DENSE)
    train_data = preprocessor.translate_label_multiclass(train_data, label_count)
    tree = treepredict.buildtree(train_data, attributes_list, label_value_list)
    treepredict.printtree(tree)

    #读取测试集，验证效果
    (test_attributes_list, test_label_value_list, test_data) = preprocessor.read_data(test_data_file, label_count, arff.DENSE)
    test_data_copy = copy.deepcopy(test_data)
    predicted_labels_list = []
    for row in test_data:
        result = treepredict.classify(row, tree, test_attributes_list)
        post_result = treepredict.post_classify(result)
        decoded_result = preprocessor.label_decoding(post_result)
        predicted_labels_list.append(decoded_result)

    hamming_loss = postprocessor.hamming_loss(test_data_copy, predicted_labels_list)
    print('hamming loss of merging labels:', hamming_loss)
else :
    #当做多个二类分类问题处理
    (attributes_list, label_value_list, train_data) = preprocessor.read_data(train_data_file, label_count, arff.DENSE)
    trees = []
    for label_index in range(0, label_count):   #建立label_count个决策树，每个决策树对应一个二类分类问题(Binary Classification)
        binary_data = preprocessor.translate_label_binary(train_data, label_count, label_index)
        print('numbers of attributes of binary_data', len(binary_data[1]))
        trees.append(treepredict.buildtree(binary_data, attributes_list, label_value_list))
        print('label index:', label_index)
        treepredict.printtree(trees[label_index])

    #用label_count个决策树，分别预测每个标签
    (test_attributes_list, test_label_value_list, test_data) = preprocessor.read_data(test_data_file, label_count, arff.DENSE)
    predicted_labels_list = []  #所有样本的标签组列表
    test_data_copy = copy.deepcopy(test_data)
    for row in test_data:
        predicted_labels = []   #每个样本数据，对应的预测标签组
        for label_index in range(0, label_count):
            single_label_row = preprocessor.translate_label_binary_line(row, label_count, label_index)
            predicted_label = treepredict.classify(single_label_row, trees[label_index], test_label_value_list)
            predicted_label = treepredict.post_classify(predicted_label)
            predicted_labels.append(predicted_label)
            print('label_index:', label_index,)
            print('predict_label:', predicted_label)
        print(predicted_labels)
        predicted_labels_list.append(predicted_labels)

    hamming_loss = postprocessor.hamming_loss(test_data_copy, predicted_labels_list)
    print('hamming loss on several binary classification:', hamming_loss)







# tree = treepredict.buildtree(treepredict.my_data)
# print('剪枝前')
# treepredict.printtree(tree)
#
# new_data = ['(direct)','USA','yes',5]
# result = treepredict.classify(new_data, tree)
# print(new_data, result)

