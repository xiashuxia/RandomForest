import treepredict
import preprocessor

def hamming_loss(test_data, predicted_label_list):
    loss = 0
    for index in range(0, len(test_data)):
        label_count = len(predicted_label_list[index])
        example = test_data[index]
        real_labels = example[(len(example) - label_count):]
        predicted_labels = predicted_label_list[index]
        print('real_labels:' + str(real_labels) + ' predicted_labels:' + str(predicted_labels))

        miss = 0    #逐一对比真实标签real_labels和预测标签predicted_labels,统计误预测个数
        for i in range(0, label_count):
            if real_labels[i] != predicted_labels[i]:
                miss += 1

        miss_percentage = float(miss) / label_count
        print('miss_percentage:', miss_percentage)
        loss += miss_percentage

    hammingloss = float(loss) / len(test_data)
    return hammingloss