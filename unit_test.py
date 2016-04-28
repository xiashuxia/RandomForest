import unittest
import treepredict
import preprocessor
import postprocessor
import os

class MyTestCase(unittest.TestCase):
    def test_divideset_with_discrete_attribute(self):
        (set1, set2) = treepredict.divideset(treepredict.my_data, 2,'yes', True)
        self.assertEqual(len(set1), 8)
        self.assertEqual(len(set2), 7)

    def test_divideset_with_continuous_attribute(self):
        (set1, set2) = treepredict.divideset(treepredict.my_data, 3, 20, False)
        self.assertEqual(len(set1), 6)

    def test_uniquecount(self):
        results = treepredict.uniquecounts(treepredict.my_data)
        self.assertEqual(results['Basic'], 6)
        self.assertEqual(results['Premium'], 3)

    def test_entropy(self):
        self.assertAlmostEqual(1,2,msg='This method isn\'t completed', delta = 0.5)

    def test_split_selection(self):
        best_criterion = treepredict.split_selection(treepredict.my_data,treepredict.my_data_attribute_list, treepredict.my_data_label_list)
        print(best_criterion)
        self.assertEqual(1, 2, msg='I don\'t know')

    def test_store_tree(self):
        tree_array = [[1, 1, None],
                      [1, 2, None],
                      ]
        preprocessor.store_tree('.\\test.test', str(tree_array))
        self.assertTrue(os.path.exists('.\\test.test'))

    def test_hamming_loss(self):
        test_data=[[1, 2, 3, '0', '1', '1', '1'],
                   [1, 2, 3, '0', '1', '1', '1'],
                   [1, 2, 3, '0', '1', '1', '1'],
                   [1, 2, 3, '0', '1', '1', '1']]
        predicted_labels_list = [['0', '0', '1', '0'],
                                 ['0', '0', '1', '0'],
                                 ['0', '0', '1', '0'],
                                 ['0', '0', '1', '0']]
        hammingloss = postprocessor.hamming_loss(test_data, predicted_labels_list)
        self.assertAlmostEqual(hammingloss, 0.5, delta = 0.01)





if __name__ == '__main__':
    unittest.main()
