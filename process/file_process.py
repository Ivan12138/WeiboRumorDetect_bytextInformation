#!/usr/bin/env python
# encoding:utf-8

"""
@Proofreading:wh
@file: file_process.py
@Description:将原文件夹的所有rumor文件,标签，并且将需要的特征写入一个csv文件
"""

from pathlib import Path
import json
import re

analysis_root_dir = '../dataset/original-microblog/rumor/'
store_result = '../dataset/rumor.csv'

# 读入目录下的所有文件，并标签为‘1’
def parse_dir(root_dir):
    path = Path(root_dir)
    all_json_file = list(path.glob('*.json'))
    parse_result = []

    for json_file in all_json_file:
        # 获取所在目录的名称
        with open(str(json_file), 'r', encoding='utf-8') as f:
            json_result = json.load(f)  # 加载每个文件
        json_result['label'] = 1  # 把每个文件加label标记为1
        parse_result.append(json_result)

    return parse_result


# 将json文件中的需要的特征写入一个文件
def write_result_in_file(write_path, write_content):
    with open(write_path, 'w', encoding='utf-8') as f:
        f.writelines('text, gender, followers, location, friends, comments, source, likes, reposts, label\n')
        for dict_content in write_content:
            # 把中文'，'替换成英文‘,'，如果不存在该符号，则继续
            try:
                text = str(dict_content['text'])
                text = text.replace(',', '，')
            except:
                continue
            print(text)
            # 若user为空，则其下的特征也为空
            if dict_content['user'] == 'empty':
                gender = 'null'
                followers = 'null'
                location = 'null'
                friends = 'null'
            else:
                # 若特征不为空，则传到相关字段
                user_dict = dict_content['user']
                try:
                    gender = user_dict['gender']
                except:
                    gender = 'null'
                print(gender)
                try:
                    followers = user_dict['followers']
                except:
                    followers = 'null'
                print(followers)
                try:
                    location = user_dict['location']
                except:
                    location = 'null'
                print(location)
                try:
                    friends = user_dict['friends']
                except:
                    friends = 'null'
                print(friends)
            try:
                comments = dict_content['comments']
            except:
                comments = 'null'
            print(comments)
            try:
                source = dict_content['source']
            except:
                source = 'null'
            print(source)
            try:
                likes = dict_content['likes']
            except:
                likes = 'null'
            print(likes)
            try:
                reposts = dict_content['reposts']
            except:
                reposts = 'null'
            print(reposts)
            label = dict_content['label']
            print(label)
            # 用','分隔开每个字段
            f.writelines(
                text + ',' + gender + ',' + str(followers) + ',' + str(location) + ',' + str(friends) + ',' +
                str(comments) + ',' + source + ',' + str(likes) + ',' + str(reposts) + ',' + str(label) + '\n'
            )


if __name__ == '__main__':
    print('process begin...')
    # 将所有rumor的json文件读入并标为‘1’
    parse_result = parse_dir(analysis_root_dir)
    print(parse_result)
    # 将label过得数据提取需要特征，写入一个json文件中
    write_result_in_file(store_result, parse_result)
    print('process finished...')
