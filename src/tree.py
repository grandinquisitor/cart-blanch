import operator
from collections import defaultdict
from math import log
import unittest
from itertools import *
import sys
import re
import datetime
from cStringIO import StringIO


# http://www.autonlab.org/tutorials/dtree.html
# some code based on Segaran, 2007

class purity:
    """
    namespace for cell purity metrics. all members are static
    """
    
    @staticmethod
    def gini(datarows):
        """
        probability that a randomly placed item will be in the wrong category
        """
    
        total = float(len(datarows))
        counts = _freq_dist(datarows)
    
        impurity = 0
    
        for k1, count1 in counts.iteritems():
            p1 = count1 / total
            for k2, count2 in counts.iteritems():
                if k1 != k2:
                    p2 = count2 / total
                    impurity += p1 * p2
    
        # call = lambda func: lambda args: func(*args)
        # filter(call(operator.__ne__), product(counts, counts))
        # map(lambda (x, y): (x/total) * (y/total), filter(call(operator.__ne__), product(counts, counts)))
        # map(lambda (x, y): (x * y) / (total ** 2), filter(call(operator.__ne__), product(counts, counts)))
        
        return impurity
    
    
    @staticmethod
    def entropy(datarows):
        """
        amount of information in each cell
        """
        
        total = float(len(datarows))
        counts = _freq_dist(datarows)
        entropy = 0.0
        for count in counts.itervalues():
            p = count / total
            entropy = entropy - p * log(p, 2)
    
        return entropy

    @staticmethod
    def variance(datarows):
        if len(datarows)==0: return 0
        data=[float(row[len(row)-1]) for row in datarows]
        mean=sum(data)/len(data)
        variance=sum([(d-mean)**2 for d in data])/len(data)
        return variance


    default = entropy
    

class decisionnode(object):
    def __init__(self, col_ix = -1, value=None, results=None, tb= None, fb= None, test_obj=None):
        self.col_ix = col_ix # column index of the criteria to be tested
        self.value = value # value that the column must match to get a true result
        self.true_branch = tb # next node if result is true
        self.false_branch = fb # next node if result is false
        
        self.test_obj = test_obj

        assert isinstance(results, (list, tuple)) or results is None, results

        if results is not None:
            self.results = _freq_dist(results)
            self.result_data = results # dictionary of results for this branch. only used by endpoints
        else:
            self.results = None
            self.result_data = None


    def resolve(self, *coords):
        """
        resolve a set of branch coordinates. useful for traversing a tree by hand
        """
        if not coords: # list is empty
            return self
        else:
            assert isinstance(coords[0], bool) or coords[0] in (0, 1), type(coords[0])
            
            if coords[0]:
                return self.true_branch.resolve(*coords[2:])
            else:
                return self.false_branch.resolve(*coords[1:])


    def results_of_class(self, classname):
        """
        return all of the results at this node of a particular class. leaf nodes only
        """
        if not self.result_data:
            raise RuntimeError, "this is not a leaf node"
        
        return [x for x in self.result_data if x['__CLASS__'] == classname]


    def printme(self, indent=''):
        """
        print an abstract representation of this classification tree
        """
        
        if self.results is not None:
            print indent, repr(dict(self.results))
        else:
            print indent, "col %s %s %s?" % (self.col_ix, isinstance(self.value, bool) and '=' or '>', self.value)

            indent += '  '
            print indent, 'yes -> '
            self.true_branch.printme(indent + '  ')
            print indent, 'no -> '
            self.false_branch.printme(indent + '  ')



    def save_mysql_classifier(self, db, name, col_key):
        swap = sys.stdout
        sys.stdout = StringIO()

        try:
            self.print_mysql_classifier(name, col_key, False)

        finally:
            output_capture = sys.stdout
            sys.stdout = swap
            sys.stdout.write(output_capture.getvalue())

        c = db.cursor()
        c.execute("drop function if exists " + name)
        c.execute(output_capture.getvalue())
        c.close()
        
        
    @staticmethod
    def _get_comment():
        return "auto-generated" + (" by '" +  sys.argv[0]  + "'" if sys.argv[0] else '') + " on " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %z')

    @staticmethod
    def _get_best(results):
        return sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]
    
    def print_python_classifier(self, name, indent='  '):
        """
        compile this classification tree as a native python function and print to stdout
        """
        
        print "def " + name + "(" + ', '.join(sorted(self.sig_columns())) + ")" + ":"
        print indent + "# " + self._get_comment()
        self._print_classifier_python_recurse(indent)
        
    def _print_classifier_python_recurse(self, start_indent, indent=''):
        indent += start_indent
        if self.results is not None:
            best = self._get_best(self.results)
            print indent + 'return', repr(best)
        
        else:
            assert self.col_ix
            if isinstance(self.test_obj, numeric_feature):
                print indent + "if %s %s %s:" % (self.test_obj.field, isinstance(self.value, bool) and '==' or '>', self.value)
            else:
                assert isinstance(self.test_obj, regex_feature)
                print indent + "if %sre.compile(%r).search(%s):" % ("" if self.value else "not ", self.test_obj.pattern, self.test_obj.field)
                
            self.true_branch._print_classifier_python_recurse(start_indent, indent)
        
            print indent + 'else:'
            self.false_branch._print_classifier_python_recurse(start_indent, indent)



    def print_mysql_classifier(self, name, delim=True):
        """
        compile this classification tree as a mysql function and print to stdout 
        """
        
        if delim:
            print "DROP FUNCTION IF EXISTS " + name + ";"
            print "DELIMITER ////"
        print "CREATE FUNCTION " + name + " (" + ', '.join(col + ' VARCHAR(255)' for col in sorted(self.sig_columns())) + ")"
        print "  RETURNS VARCHAR(255)"
        print "  DETERMINISTIC"
        print "  NO SQL"
        print "  SQL SECURITY INVOKER"
        print "  -- " + self._get_comment()
        print "  BEGIN"
        self._print_classifier_mysql_recurse(' ')
        print "END"
        if delim:
            print "////"
            print "DELIMITER ;"


    def _print_classifier_mysql_recurse(self, indent=''):
        if self.results is not None:
            best = self._get_best(self.results)
            print indent, 'RETURN', repr(best), ";"
        else:
            assert self.col_ix
            if isinstance(self.test_obj, numeric_feature):
                print indent, "IF %s %s %s THEN " % (self.test_obj.field, isinstance(self.value, bool) and '=' or '>', self.value)
            else:
                print indent, "IF %s REGEXP %r = %s THEN " % (self.test_obj.field, self.test_obj.pattern, int(self.value))
 
            self.true_branch._print_classifier_mysql_recurse(indent+'  ')
            print indent, "ELSE"
            self.false_branch._print_classifier_mysql_recurse(indent+'  ')
            print indent, "END IF;"



    def prune(self, mingain, spec_func=purity.entropy):
        """
        prune the tree.
        
        i.e. to prevent overfitting.
        """
        if self.true_branch.results is None:
            self.true_branch.prune(mingain, spec_func)

        if self.false_branch.results is None:
            self.false_branch.prune(mingain, spec_func)

        if None not in (self.true_branch.results, self.false_branch.results):
            new_tb, new_fb = [], []

            if self.true_branch.result_data:
                new_tb += self.true_branch.result_data
            else:
                for v, c in self.true_branch.results.iteritems():
                    new_tb += [{'__CLASS__': v}] * c

            if self.false_branch.result_data:
                new_fb += self.false_branch.result_data
            else:
                for v, c in self.false_branch.results.iteritems():
                    new_fb += [{'__CLASS__': v}] * c

            # uh, is the below correct?
            delta = spec_func(new_tb + new_fb) - (spec_func(new_tb) + spec_func(new_fb)/2)
            
            if delta < mingain:
                self.results = _freq_dist(new_tb + new_fb)
                self.result_data = new_tb + new_fb
                self.true_branch = None
                self.false_branch = None
                # print "merging"

    def classify(self, observation):
        """
        classify a single observation row.
        """
        
        result = self._classify_recurse(observation)
        return self._get_best(result)


    def _classify_recurse(self, observation):
        if self.results is not None:
            return dict(self.results)

        else:
            v = observation[self.col_ix]
            
            result = self._get_op(self.value)(self.value, v)
            chosen_branch = {True: self.true_branch, False: self.false_branch}[result]

            # print self.value, ' is ', get_op(self.value), 'of', v, result

            return chosen_branch.classify_recurse(observation)

    def sig_tests(self):
        """
        return the set of tests that were useful in classification
        """
        
        if self.results is None:
            return set((self.col_ix,)) | self.true_branch.sig_tests() | self.false_branch.sig_tests()
        else:
            return set()

    def sig_columns(self):
        """
        return the set of fields that had tests that were useful in classification
        """
        if self.results is None:
            return set((self.test_obj.field,)) | self.true_branch.sig_columns() | self.false_branch.sig_columns()
        else:
            return set()
        

    def expected_error(self, _start=True):
        """
        determine the expected error of this model based on the training set (a.k.a. "training set error").
        
        this is essentially using the training set as a test set. for accuracy subtract the result of this function from one.
        """
        
        if (_start):
            result = map(float, self.expected_error(False))
            return result[0] / (result[1] + result[0])
            # error / (accuracy + error)
        
        else:
            if self.results is not None:
                # return (number of expected errors, total in this node)
                total = sum(self.results.values())
                return (total - max(self.results.values()), total)
    
            else:
                assert self.true_branch and self.false_branch
                (err1, true1) = self.true_branch.expected_error(False)
                (err2, true2) = self.false_branch.expected_error(False)
                return (err1 + err2, true1 + true2)
                #return map(sum, zip(branch.expected_errors() for branch in (self.true_branch, self.false_branch)))



    def test_accuracy(self, rows, verbose=False):
        """
        assess accuracy based on many input rows. each row must contain an item named '__CLASS__' that contains the correct classification label
        
        set verbose=True if you want the result of every row printed to stdout
        """
        
        correct = 0
        incorrect = 0
        for row in rows:
            correct_label = row['__CLASS__']
            determined_label = self.classify(row)
            if correct_label == determined_label:
                correct += 1
            else:
                incorrect += 1
            
            if verbose:
                print repr(row), correct_label, determined_label
                
        if (correct or incorrect):
            return float(correct) / (incorrect + correct)
        else:
            return None


    @staticmethod
    def _get_op(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):  #operator.isNumberType(value):
            return operator.__gt__
        else:
            return operator.__eq__
    
    @classmethod
    def _split_set(cls, rows, column, value):
        # split the 
        split_func = cls._get_op(value)
            
        return (
            [row for row in rows if split_func(row[column], value)],
            [row for row in rows if not split_func(row[column], value)]
            )




def _freq_dist(rows):
    # return the frequency distribution of each outcome (named '__CLASS__') in a list of dicts
    
    results = defaultdict(int)
    for row in rows:
        try:
            results[row['__CLASS__']] += 1
        except:
            print row
            raise
    return results






class feature(object):
    # abstract base class
    pass


class numeric_feature(feature):
    
    "a numeric discriminator"

    def __init__(self, name, field=None):
        self.name = name
        self.field = field or name

    def evaluate(self, input):
        return input[self.field]



class regex_feature(feature):

    "a regex discriminator"

    default_field = None

    def __init__(self, name, pattern, field=None, should_match=(), dont_match=()):
        self.name = name
        self.pattern = pattern
        self.field = field or self.default_field
        assert isinstance(should_match, tuple)
        assert isinstance(dont_match, tuple)
        self.positive_matches = should_match
        self.negative_matches = dont_match

    def __str__(self):
        return self.pattern

    def evaluate(self, inputstrs):
        assert isinstance(inputstrs, dict)
        return bool(re.compile(self.pattern, re.I).search(inputstrs[self.field]))


    def add_tests(self, test_suite, db=None):
        "make a unit test and add it to the supplied test suite"

        i = 0
        for i, (exp_result, input_str) in enumerate(chain(izip(cycle((True,)), self.positive_matches), izip(cycle((False,)), self.negative_matches))):
            class anon_py_test(unittest.TestCase):
                def runTest(self):
                    self.assertEqual(bool(re.compile(self.pattern, re.I).search(self.input_str)), self.exp_result, "%s (re: %s) got %s against %s" % (self.name, self.pattern, not self.exp_result, self.input_str))

            anon_py_test.pattern = self.pattern
            anon_py_test.name = self.name
            anon_py_test.input_str = input_str
            anon_py_test.exp_result = exp_result
            anon_py_test.__name__ = 'feature_test_py_%s_%03d' % (self.name, i)

            test_suite.addTest(anon_py_test())


            if db:
                class anon_my_test(unittest.TestCase):
                    def runTest(self):
                        c = db.cursor()
                        c.execute("select %s REGEXP %s", (self.input_str, self.pattern))
                        self.assertEqual(bool(c.fetchone()[0]), self.exp_result, "%s (REGEXP: %s) got %s against %s `%s`" % (self.name, self.pattern, not self.exp_result, self.input_str, c._executed))
                        c.close()

                anon_my_test.pattern = self.pattern
                anon_my_test.name = self.name
                anon_my_test.input_str = input_str
                anon_my_test.exp_result = exp_result
                anon_my_test.__name__ = 'feature_test_my_%s_%03d' % (self.name, i)

                test_suite.addTest(anon_my_test())


        # if no tests defined, just make sure that the pattern compiles without throwing an error
        if not i:
            class anon_py_test(unittest.TestCase):
                def runTest(self):
                    re.compile(self.pattern)

            anon_py_test.pattern = self.pattern
            anon_py_test.name = self.name
            anon_py_test.__name__ = 'discrim_test_my_%s_baseline' % (self.name,)

            test_suite.addTest(anon_py_test())

            if db:
                class anon_my_test(unittest.TestCase):
                    def runTest(self):
                        c = db.cursor()
                        c.execute("select '' REGEXP %s", (self.pattern,))

                anon_my_test.pattern = self.pattern
                anon_my_test.name = self.name
                anon_my_test.__name__ = 'discrim_test_my_%s_baseline' % (self.name,)

                test_suite.addTest(anon_my_test())

            

class feature_collection(dict):
    "collection of tests. useful for auto-generating unit tests"

    def __init__(self, *args, **kwargs):
        default_field = kwargs.get("default_field")
        
        self.column_names = set()
        
        for arg in args:
            if arg.name in self: raise ValueError("%s already exists" % arg.name)
            self[arg.name] = arg
            
            if default_field is not None and arg.field is None:
                arg.field = default_field
            
            assert arg.field is not None
            self.column_names.add(arg.field)


    def get_test_suite(self, dbconn=None):
        all_tests = unittest.TestSuite()
        for item in self.itervalues():
            if hasattr(item, "add_tests"):
                item.add_tests(all_tests, dbconn)
        return all_tests


    def build_training_set(self, input_rows, label_field='__CLASS__'):
        # label_field: which field in each input row specifies class label
        
        output_rows = []
        
        for input_row in input_rows:
            assert '__CLASS__' not in input_row or label_field == '__CLASS__'
            assert '__DATA__' not in input_row

            output_row = {'__CLASS__': input_row[label_field], '__DATA__': input_row}

            for test_name, test_obj in self.iteritems():
                output_row[test_name] = test_obj.evaluate(input_row)

            output_rows.append(output_row)

        return output_rows
    
    
    def buildtree(self, training_set, score_func=purity.entropy):
        "build a classification tree from a training set"
    
        assert callable(score_func)
    
        if not training_set:
            return decisionnode()
    
        current_score = score_func(training_set)
    
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        
        for col_ix in self.iterkeys():
            if not col_ix.startswith('_'): # ignore any column that starts with an underscore, this includes __CLASS__
    
                column_values = set(r[col_ix] for r in training_set)
    
                for value in column_values:
                    (set1, set2) = decisionnode._split_set(training_set, col_ix, value)
    
                    # information_gained
                    p = float(len(set1)) / len(self)
                    if set1 and set2:
                        gain = current_score - p * score_func(set1) - (1-p) * score_func(set2)
                        if gain > best_gain:
                            best_gain = gain
                            best_criteria = (col_ix, value)
                            best_sets = (set1, set2)
    
        if best_gain > 0:
            true_branch = self.buildtree(best_sets[0])
            false_branch = self.buildtree(best_sets[1])
            return decisionnode(col_ix=best_criteria[0], value=best_criteria[1], tb=true_branch, fb=false_branch, test_obj=self[best_criteria[0]])
        else:
            return decisionnode(results=training_set)
