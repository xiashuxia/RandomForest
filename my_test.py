import preprocessor
import treepredict
import postprocessor
import arff



#读取训练集，建树(多标签转换成单标签)
label_count = 6
# (attributes_list, label_list,train_data) = preprocessor.read_data('.\\scene\\scene-train-tiny.arff',
#                                                                   label_count, arff.DENSE)
# train_data = preprocessor.translate_label_multiclass(train_data, label_count)
# tree = treepredict.buildtree(train_data, attributes_list, label_list)
# treepredict.printtree(tree)
#
# #测试决策树文件读写
# tree_list = preprocessor.tree2array(tree)
# preprocessor.store_tree('.\\my_tree', tree_list)

#从文件中加载决策树
loaded_tree_list = preprocessor.load_tree('.\\my_tree')
loaded_tree = preprocessor.list2tree(loaded_tree_list)


#读取测试集，验证效果
(test_attributes_list, test_label_value_list, test_data) = preprocessor.read_data('.\\scene\\scene-test-tiny.arff',
                                                                      label_count, arff.DENSE)

results = []
for row in test_data:
    result = treepredict.classify(row, loaded_tree, test_label_value_list)
    print('predict result:', result, 'test case', row)
    post_result = treepredict.post_classify(result)
    results.append(preprocessor.label_decoding(post_result))
hammingloss = postprocessor.hamming_loss(test_data, results)
print('hamming loss:', hammingloss)

