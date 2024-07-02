import ast
import os
from typing import Dict, List
import pandas as pd
import recognizers_suite
from recognizers_suite import Culture
from sqlalchemy import create_engine
from pandasql import sqldf
import re
import multiprocessing

## Adapted from Binder and Dater paper 
culture = Culture.English
def dict2df(table: list) -> pd.DataFrame:
    """
    Dict to pd.DataFrame.
    tapex format.
    """
    header, rows = table[0], table[1:]
    # print('header before : ', header)
    header = preprocess_columns(header)
    # print('header after: ', header)
    df = pd.DataFrame(data=rows, columns=header)
    return df

def preprocess_columns(columns):
    # columns = table.split('\n')[0].split('|')
    # print('preprocessing columns')
    tab_coll = []
    illegal_chars_1 = [' ', '/', '\\', '-', ':', '#', '%']
    illegal_chars_2 = ['.', '(', ')', '[', ']', '{', '}', '*', '$', ',', '?', '!', '\'', '$', '@', '&', '=',
                       '+']
    for x in columns:
        x = x.strip()
        # print(x)
        if x.isnumeric():
            x = "_" + x
        x = x.replace(">", "GT")
        x = x.replace("<", "LT")
        x = x.replace("\\n", "_")
        x = x.replace("\n", "_")
        x = x.replace('\\', '_')
        for char in illegal_chars_1:
            x = x.replace(char, '_')
        for char in illegal_chars_2:
            x = x.replace(char, '')
        tab_coll.append(x.strip())

    counts = {}
    preprocessed_colls = []
    for item in tab_coll:
        if item == '':
            item = 'column'
        if item in counts:
            counts[item] += 1
            preprocessed_colls.append(f"{item}{counts[item]}")
        else:
            counts[item] = 0
            preprocessed_colls.append(item)

    return preprocessed_colls

def convert_df_type(df: pd.DataFrame, lower_case=True):
    """
    A simple converter of dataframe data type from string to int/float/datetime.
    """

    def get_table_content_in_column(table):
        if isinstance(table, pd.DataFrame):
            header = table.columns.tolist()
            rows = table.values.tolist()
        else:
            # Standard table dict format
            header, rows = table['header'], table['rows']
        all_col_values = []
        for i in range(len(header)):
            one_col_values = []
            for _row in rows:
                one_col_values.append(_row[i])
            all_col_values.append(one_col_values)
        return all_col_values

    # Rename empty columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header == '':
            new_columns.append('FilledColumnName')  # Fixme: give it a better name when all finished!
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Rename duplicate columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header in new_columns:
            new_header, suffix = header, 2
            while new_header in new_columns:
                new_header = header + '_' + str(suffix)
                suffix += 1
            new_columns.append(new_header)
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Recognize null values like "-"
    null_tokens = ['', '-', '/']
    for header in df.columns:
        df[header] = df[header].map(lambda x: str(None) if x in null_tokens else x)

    # Convert the null values in digit column to "NaN"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_number_flag = True
        for row_i, cell_value in enumerate(one_col_values):
            try:
                float(cell_value)
            except Exception as e:
                if not cell_value in [str(None), str(None).lower()]:
                    # None or none
                    all_number_flag = False
        if all_number_flag:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: "NaN" if x in [str(None), str(None).lower()] else x)

    # Normalize cell values.
    for header in df.columns:
        df[header] = df[header].map(lambda x: str_normalize(x))

    # Strip the mis-added "01-01 00:00:00"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_with_00_00_00 = True
        all_with_01_00_00_00 = True
        all_with_01_01_00_00_00 = True
        for row_i, cell_value in enumerate(one_col_values):
            if not str(cell_value).endswith(" 00:00:00"):
                all_with_00_00_00 = False
            if not str(cell_value).endswith("-01 00:00:00"):
                all_with_01_00_00_00 = False
            if not str(cell_value).endswith("-01-01 00:00:00"):
                all_with_01_01_00_00_00 = False
        if all_with_01_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01-01 00:00:00")])
            continue

        if all_with_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01 00:00:00")])
            continue

        if all_with_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len(" 00:00:00")])
            continue

    # Do header and cell value lower case
    if lower_case:
        new_columns = []
        for header in df.columns:
            lower_header = str(header).lower()
            if lower_header in new_columns:
                new_header, suffix = lower_header, 2
                while new_header in new_columns:
                    new_header = lower_header + '-' + str(suffix)
                    suffix += 1
                new_columns.append(new_header)
            else:
                new_columns.append(lower_header)
        df.columns = new_columns
        for header in df.columns:
            # df[header] = df[header].map(lambda x: str(x).lower())
            df[header] = df[header].map(lambda x: str(x).lower().strip())

    # Recognize header type
    for header in df.columns:

        float_able = False
        int_able = False
        datetime_able = False

        # Recognize int & float type
        try:
            df[header].astype("float")
            float_able = True
        except:
            pass

        if float_able:
            try:
                if all(df[header].astype("float") == df[header].astype(int)):
                    int_able = True
            except:
                pass

        if float_able:
            if int_able:
                df[header] = df[header].astype(int)
            else:
                df[header] = df[header].astype(float)

        # Recognize datetime type
        try:
            df[header].astype("datetime64")
            datetime_able = True
        except:
            pass

        if datetime_able:
            df[header] = df[header].astype("datetime64")

    return df



def str_normalize(user_input, recognition_types=None):
    """A string normalizer which recognize and normalize value based on recognizers_suite"""
    user_input = str(user_input)
    user_input = user_input.replace("\\n", "; ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        assert len(strs_to_replace) == len(idx_pairs)
        last_end = 0
        to_concat = []
        for idx_pair, str_to_replace in zip(idx_pairs, strs_to_replace):
            to_concat.append(orig_str[last_end:idx_pair[0]])
            to_concat.append(str_to_replace)
            last_end = idx_pair[1]
        to_concat.append(orig_str[last_end:])
        return ''.join(to_concat)

    if recognition_types is None:
        recognition_types = ["datetime",
                             "number",
                             # "ordinal",
                             # "percentage",
                             # "age",
                             # "currency",
                             # "dimension",
                             # "temperature",
                             ]

    for recognition_type in recognition_types:
        if re.match("\d+/\d+", user_input):
            # avoid calculating str as 1991/92
            continue
        recognized_list = getattr(recognizers_suite, "recognize_{}".format(recognition_type))(user_input,
                                                                                              culture)  # may match multiple parts
        strs_to_replace = []
        idx_pairs = []
        for recognized in recognized_list:
            if not recognition_type == 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):
                    # if the datetime is a period:
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # in some cases, this variable could be none.
                    if len(recognized.resolution['values']) == 1:
                        strs_to_replace.append(
                            recognized.resolution['values'][0]['timex'])  # We use timex as normalization
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match("(.*)-(.*)-(.*) 00:00:00", user_input):
        user_input = user_input[:-len("00:00:00") - 1]
        # '2008-04-13 00:00:00' -> '2008-04-13'
    return user_input

def table_linearization(table: pd.DataFrame, style: str = 'pipe'):
    """
    linearization table according to format.
    """
    assert style in ['pipe', 'row_col']
    linear_table = ''
    if style == 'pipe':
        header = ' | '.join(table.columns) + '\n'
        linear_table += header
        rows = table.values.tolist()
        # print('header: ', linear_table)
        # print(rows)
        for row_idx, row in enumerate(rows):
            # print(row)
            line = ' | '.join(str(v) for v in row)
            # print('line: ', line)
            if row_idx != len(rows) - 1:
                line += '\n'
            linear_table += line

    elif style == 'row_col':
        header = 'col : ' + ' | '.join(table.columns) + '\n'
        linear_table += header
        rows = table.values.tolist()
        for row_idx, row in enumerate(rows):
            line = 'row {} : '.format(row_idx + 1) + ' | '.join(row)
            if row_idx != len(rows) - 1:
                line += '\n'
            linear_table += line
    return linear_table

def prepare_table(entry):
    table = entry['table_text']
    T = dict2df(table)
    T = T.assign(row_id=range(len(T)))
    row_id = T.pop('row_id')
    T.insert(0, 'row_id', row_id)
    col = T.columns

                # print('Table Coll: ', col)
    tab_col = ""
    for c in col:
        tab_col += c + ", "
        tab_col = tab_col.strip().strip(',')
        #print('Table Column: ', tab_col)

    engine = create_engine('sqlite:///database.db')
    w = convert_df_type(T)
    entry['table'] = w
    return entry


def normalize_all_tables(data):
    ### entry ['table_text'] = prepare_table(entry['table_text'])
    ### use multiprocessing
    pool = multiprocessing.Pool(32)
    data = pool.map(prepare_table, data)
    pool.close()
    pool.join()
    return data


if __name__ == '__main__':
    test_table =[
            [
                "",
                "live births per year",
                "deaths per year",
                "natural change per year",
                "cbr*",
                "",
                "nc*",
                "tfr*",
                "imr*"
            ],
            [
                "1950-1955",
                "139 000",
                "66 000",
                "74 000",
                "52.6",
                "24.8",
                "27.8",
                "6.86",
                "174"
            ],
            [
                "1955-1960",
                "164 000",
                "76 000",
                "88 000",
                "53.8",
                "24.9",
                "29.0",
                "6.96",
                "171"
            ],
            [
                "1960-1965",
                "195 000",
                "89 000",
                "105 000",
                "55.5",
                "25.5",
                "30.1",
                "7.13",
                "167"
            ],
            [
                "1965-1970",
                "229 000",
                "105 000",
                "124 000",
                "56.2",
                "25.8",
                "30.4",
                "7.32",
                "164"
            ],
            [
                "1970-1975",
                "263 000",
                "121 000",
                "142 000",
                "55.8",
                "25.6",
                "30.2",
                "7.52",
                "162"
            ],
            [
                "1975-1980",
                "301 000",
                "138 000",
                "164 000",
                "55.1",
                "25.1",
                "29.9",
                "7.63",
                "161"
            ],
            [
                "1980-1985",
                "350 000",
                "157 000",
                "193 000",
                "55.4",
                "24.8",
                "30.6",
                "7.76",
                "159"
            ],
            [
                "1985-1990",
                "406 000",
                "179 000",
                "227 000",
                "55.9",
                "24.6",
                "31.3",
                "7.81",
                "155"
            ],
            [
                "1990-1995",
                "471 000",
                "192 000",
                "279 000",
                "55.5",
                "22.7",
                "32.8",
                "7.78",
                "146"
            ],
            [
                "1995-2000",
                "538 000",
                "194 000",
                "344 000",
                "53.5",
                "19.3",
                "34.2",
                "7.60",
                "131"
            ],
            [
                "2000-2005",
                "614 000",
                "194 000",
                "420 000",
                "51.3",
                "16.2",
                "35.1",
                "7.40",
                "113"
            ],
            [
                "2005-2010",
                "705 000",
                "196 000",
                "509 000",
                "49.5",
                "13.8",
                "35.7",
                "7.19",
                "96"
            ]
        ]
    test = {
        'table_text': test_table
    }
    w = prepare_table(test)
    print(w)
    print(table_linearization(w['table'], style='pipe'))