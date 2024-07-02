import json
import sys
import glob
import argparse
import re
import numpy as np
import string
import sys, os, re, argparse
import unicodedata
from codecs import open
from math import isnan, isinf
from abc import ABCMeta, abstractmethod
from eval_binder import Evaluator
def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x

def maybe_normalize_float(span: str):
    if span and (re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
                 or (re.match(r"^[0-9]*[.]?[0-9]*$", span))) and span != '.':
        # FIXME: We did this(instead of try except) to convert a string into a float
        #  since the try catch will lead to an error when using 8 V100 gpus with cuda 11.0,
        #  and we still don't know why that could happen....
        return str(float(span))
    else:
        return span


def maybe_normalize_number(text: str) -> str:
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    for index, unit in enumerate(units):
        if text == unit:
            return str(float(index))
    return text


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def remove_articles(text: str) -> str:
    return re.sub(r'\b(a|an|the)\b', ' ', text)
class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None
################ Value Instantiation ################

def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                        in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))

def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True
def date_normalize(text):
    ## transform date as xxxx-x(x)-x(x)
    month2digit = {'january':1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6, 'july':7, 'august':8, 'september':9, 'october':10, 'november':11, 'december':12}
    ## example:March 9, 2016
    ## example: March 9 2016
    if re.match(r'[a-zA-Z]+ \d{1,2},? \d{4}', text):
        month, day, year = text.lower().split(' ')
        month = month
        day = day.replace(',', '')
        month = month2digit[month]
        return f'{year}-{month}-{day}'
    ## 9 March, 2015
    elif re.match(r'\d{1,2} [a-zA-Z]+,? \d{4}', text):
        day, month, year = text.lower().split(' ')
        month = month
        day = day.replace(',', '')
        month = month2digit[month]
        return f'{year}-{month}-{day}'
    else:
        return text

def eval_ex_match(pred, gold_result, question):
    pred = pred.lower()
    if ' and ' in pred:
        pred = pred.replace(' and ', ', ')
        gold_result = [g.replace(' and ', ', ') for g in gold_result]
    for i in range(len(gold_result)):
        ## if only digits and comma, remove comma
        if gold_result[i].replace(',', '').isdigit():
            gold_result[i] = gold_result[i].replace(',', '')


    if len(gold_result) > 1:
        pred = [span.strip() for span in pred.split(',')]    
    else:
        try:
            gold_result[0] = date_normalize(gold_result[0])
        except:
            pass
        pred = [pred.strip()]
    eval = Evaluator()
    return eval.eval_ex_match(pred, gold_result, 'wikitq', question=question)


import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str)
parser.add_argument('--benchmark', type = str)
parser.add_argument('--one_step', action = 'store_true')
parser.add_argument('--llm', action = 'store_true')
parser.add_argument('--debug', action = 'store_true')
args = parser.parse_args()
def main():

    st2id = {}
    import json


    path = f'data/gpt_output/{args.model_name}/{args.benchmark}_reason_parsed.json'
    if args.one_step:
        path = f'data/gpt_output/{args.model_name}/{args.benchmark}_one_step_result_parsed.json'
    if args.llm:
        path = f'data/gpt_output/{args.model_name}/{args.benchmark}_llm_result_parsed.json'
    with open(path, 'r') as f:
        data = json.load(f)
    # load json saved as a dict

    correct = 0
    wrong = 0
    if args.benchmark == 'feverous':
        for entry in data:
            if entry['result'] == 'Failed SQL':
                wrong += 1
                continue
            if entry['result'].lower() == entry['answer'].lower():
                correct += 1
            else:
                wrong += 1
                print(entry['ids'])
                print(entry['result'])
                print(entry['answer'])
                print('-------------')

    if args.benchmark == 'tabfact_small':
        for entry in data:
            if entry['result'] == 'Failed SQL':
                wrong += 1
                continue
            if entry['result'].lower() == 'true' and entry['label'] == 1:
                correct += 1
            
            elif entry['result'].lower() == 'false' and entry['label'] == 0:
                correct += 1

            else:
                wrong += 1
                print(entry['ids'])
                print(entry['result'])
                print(entry['label'])
                print('-------------')
            
    if args.benchmark == 'wikitab':
        for line in data:
            if line['result'] == 'Failed SQL':
                wrong += 1
                continue
            pred = line['result']
            ## extract answer  <1> -> 1
            pred = re.sub(r'<(\d+)>', r'\1', pred)
        #pred = ','.join(pred)
            gold = line['answer']
            #gold = ','.join(gold)
            if eval_ex_match(pred, gold, line['question']):
                correct += 1
                line['correct'] = True
            else:
                wrong += 1
                print(line['ids'])
                print(pred)
                print(gold)
                print('-------------')
                line['correct'] = False
        
    with open(path, 'w') as f:
        json.dump(data, f, indent = 4)
    print('Denotation Accuracy', correct / (correct + wrong))
    print('Corect: ', correct, 'Wrong: ', wrong, "Total: ", (correct+wrong))

if __name__ == '__main__':
    main()

#####
# pred = '1992-12-12'
# gold = ['12 December 1992']
#print(eval_ex_match(pred, gold))
#pred = '1998-2-22'
#gold = ['February 22, 1998']  