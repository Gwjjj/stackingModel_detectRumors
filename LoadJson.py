import os
import json
import re
import pickle
json_dir = 'C:\\Users\\Gwjjj\\Desktop\\rumdect\\Weibo'
global start
start = 0
# dict{eid: [cid]}
eid_childid = {}
# 微博文本
doc_list = []
# 停用词
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#     return stopwords
# stopwords = stopwordslist('C:\\Users\\Gwjjj\\Downloads\\stopword.txt')

# 主微博eid对应的label
with open('C:\\Users\\Gwjjj\\Desktop\\rumdect\\Weibo.txt', 'r') as f:
    idLable_dict = {}
    for line in f.readlines():
        blog_cast = line.split()
        eid = blog_cast[0][4:]
        label = blog_cast[1][-1]
        idLable_dict[eid] = label


# 返回文件夹的文件列表
def get_fileroute(route):
    file_route_list = []
    for file in sorted(os.listdir(route)):
        file_rounte = os.path.join(route, file)
        file_route_list.append(file_rounte)
    return file_route_list


# 提取.json格式的内容
def get_json(file_route):
    if file_route.endswith('.json'):
        json_file = open(file_route, encoding='utf-8')  # 默认以gbk模式读取文件，当文件中包含中文时，会报错
        json_body = json.load(json_file)
        return json_body
    else:
        return None


def get_dict_idText(jsons, ee):
    global start
    # label = [0, 1] # 谣言
    # label = [1, 0] # 非谣言
    blog_idList = []
    eid = start
    if idLable_dict[ee] == '0':
        for js in jsons:
            blog_text = js['text']
            blog_text = re.sub('[a-zA-Z0-9_/:.转发微博。&轉發]', '', blog_text).strip()
            if blog_text:
                blog_idList.append(start)
                doc_list.append(blog_text)
                start += 1
        eid_childid[eid] = blog_idList
    

dir_list = get_fileroute(json_dir)
i = 0
length = len(dir_list)
for i, single_file in enumerate(dir_list[:5]):
    print("还剩", length - i - 1)
    get_dict_idText(get_json(single_file), single_file[37:][:-5])

# write_eid_childid = open('norumor_eid_childid.pkl', 'wb')
# write_docList = open('norumor_docList.pkl', 'wb')
# pickle.dump(eid_childid, write_eid_childid)
# pickle.dump(doc_list, write_docList)
# print(eid_childid)
# print(start)
# print(len(eid_childid))
# print(len(doc_list))