my_data = [['slashdot','USA','yes',18,'None'],
           ['google','France','yes',23,'Premium'],
           ['digg','USA','yes',24,'Basic'],
           ['kiwitobes','France','yes',23,'Basic'],
           ['google','UK','no',21,'Premium'],
           ['(direct)','New Zealand','no',12,'None'],
           ['(direct)','UK','no',21,'Basic'],
           ['google','USA','no',24,'Premium'],
           ['slashdot','France','yes',19,'None'],
           ['digg','USA','no',18,'None'],
           ['google','UK','no',18,'None'],
           ['kiwitobes','UK','no',19,'None'],
           ['digg','New Zealand','yes',12,'Basic'],
           ['google','UK','yes',18,'Basic'],
           ['kiwitobes','France','yes',19,'Basic']]
my_data_attribute_list = [
    ('source',['slashdot', 'google', 'digg', 'kiwitobes', '(direct)']),
    ('location', ['USA', 'France', 'UK', 'New Zealand']),
    ('FAQ', ['yes', 'no']),
    ('pages', 'numeric')
]

my_data_label_list = [
    ('None', ['0', '1']),
    ('Basic', ['0', '1']),
    ('Premium', ['0', '1'])
]

#定义决策树
class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results #仅当节点为叶节点时，results不为None
        self.tb =tb
        self.fb = fb

#
def divideset(rows,column,value,discrete=True):
    if not discrete:
        split_function = lambda row:row[column] >= value
    else:
        split_function = lambda row:row[column] == value

    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

#统计每种标签的数量
def uniquecounts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results

#计算集合rows的熵entropy = p(x) * log2(p(x))
def entropy(results):
    from math import log
    log2 = lambda x:log(x)/log(2)

    #计算样本总数
    count = 0
    for r in results.keys():
        count += results[r]

    # 计算熵
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / count
        ent = ent - p * log2(p)
    return ent


""" 寻找最佳分割属性及其split point.
Args：
    rows: 样本列表
    scoref: 不纯度度量指标

Return:
    best_criterion: 一个记录最佳分割点的truple,若信息增益小于零，则best_criterion=()
        best_criterion[0]为第几列属性作为分割属性，
        best_criterion[1]为分割值，
        best_cirterion[2]为属性是否离散值.
"""
def split_selection(rows, attribute_list, label_list, scoref=entropy):
    #统计共有几个属性
    attribute_count = len(attribute_list)
    current_score = entropy(uniquecounts(rows))

    best_gain = 0.0
    best_criterion = None

    #对每个属性尝试拆分
    for col in range(0, attribute_count):
        #统计第col列的属性中，每种取值的个数(先假定是离散的)
        attribute_values = {}
        for row in rows:
            if row[col] not in attribute_values: attribute_values[row[col]] = 0
            attribute_values[row[col]] += 1

        discrete = False  # 检测是否为离散属性
        if isinstance(attribute_list[col][1], (list, set)): discrete = True

        #寻找拆分条件
        if discrete:    #第col列属性为离散值
            for value in attribute_values.keys():
                (set1, set2) = divideset(rows, col, value, discrete)
                #计算信息增益，找出信息增益最大时的分割准则
                p = float(len(set1)) / len(rows)
                gain = current_score - p * scoref(uniquecounts(set1)) - (1 - p) * scoref(uniquecounts(set2))
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criterion = (col, value, True)
        else:   #第col列属性是连续值
            #所有连续的取值，为了计算每对相邻值得中点
            attribute_value_list = list(attribute_values.keys())
            for i in range(0, len(attribute_value_list)):   #把字符串转换成连续型数值
                attribute_value_list[i] = float(attribute_value_list[i])
            middle_points = []  #存储中点
            for i in range(0, len(attribute_value_list) - 1):
                middle_points.append((attribute_value_list[i] + attribute_value_list[i + 1]) / 2)
                #尝试每个中点值作为split value
                (set1, set2) = divideset(rows, col, middle_points[i], discrete)
                p = float(len(set1)) / len(rows)
                gain = current_score - p * scoref(uniquecounts(set1)) - (1 - p) * entropy(uniquecounts(set2))
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criterion = (col, middle_points[i], False)
    #如果信息增益不大于零，则返回空truple
    if best_gain <= 0:
        return ()
    else :
        return best_criterion


#scoref是度量集合不纯度的指标，这里采用基本的熵entropy
#label_list暂时不使用
def buildtree(rows, attribute_list, label_list, scoref=entropy):
    if len(rows) == 0:return decisionnode()

    best_criterion = split_selection(rows, attribute_list, label_list,scoref)

    #创建分支,拆分后，best_criterion不为空，也就是说，存在一种分割，是的信息增益为正
    if len(best_criterion) != 0:
        (set1, set2) = divideset(rows, best_criterion[0], best_criterion[1], best_criterion[2])
        best_sets = (set1, set2)
        trueBranch = buildtree(best_sets[0], attribute_list, label_list)
        falseBranch = buildtree(best_sets[1], attribute_list, label_list)
        return decisionnode(col = best_criterion[0],value = best_criterion[1], tb = trueBranch, fb = falseBranch)

    else:
        return decisionnode(results = uniquecounts(rows))

def printtree(tree, indent=''):
    #如果是叶节点
    if tree.results != None:
        print(str(tree.results))
    else:
        #打印判断条件
        print(str(tree.col) + ':' + str(tree.value) + '?')
        #打印分支
        print(indent + 'T->', end = ' ')
        printtree(tree.tb, indent + ' ')
        print(indent + 'F->', end = ' ')
        printtree(tree.fb, indent + ' ')

#预测未知数据
def classify(observation, tree, label_value_list) :
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        label_values = label_value_list[tree.col][1]
        if label_values == 'numeric' or label_values == 'real':
            if v >= tree.value: branch = tree.tb
            else: branch = tree.fb
        else:
            if v == tree.value: branch = tree.tb
            else: branch = tree.fb
        return classify(observation, branch, label_value_list)

def post_classify(predicted_results):
    label = None
    count = 0
    for key in predicted_results.keys():
        if predicted_results[key] > count:
            label = key
            count = predicted_results[key]

    return label

#剪枝
def prune(tree, mingain):
    #节点不为叶节点是，考虑对其进行剪枝，即，若子节点能合并到父节点
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)

    #如果两个分支都是叶节点，则判断他们是否需要合并
    if tree.tb.results != None and tree.fb.results != None:
        #计算左右子树的样本数量
        tcount = 0
        fcount = 0
        for r in tree.tb.results.keys():
            tcount += tree.tb.results[r]
        for r in tree.fb.results.keys():
            fcount += tree.fb.results[r]
        count = tcount + fcount

        #将左右子树的样本合并到一个节点
        results = tree.tb.results
        for r in tree.fb.results.keys():
            if r not in results:
                results[r] = tree.fb.results[r]
            else:
                results[r] += tree.fb.results[r]

        #计算在当前节点分割时，其信息增益，并且与最小增益mingain对比
        delta = entropy(results) - entropy(tree.tb.results) * (float(tcount) / count) - entropy(tree.fb.results) * (float(fcount) / count)
        if delta < mingain:
            #合并分支
            tree.tb,tree.fb = None, None
            tree.results = results



