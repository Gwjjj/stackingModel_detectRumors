import requests
import json
import argparse
import sys
import re
from requests.packages.urllib3.exceptions import InsecureRequestWarning
 
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


headers = {
'Connection':'keep-alive',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36',
}

def getWeibo(weiboId,url):
    print(url)
    response = requests.get(url=url, headers=headers, verify=False)
    reptext = response.text
    print(reptext)
    jsonobj = json.loads(reptext)
    nextid = jsonobj['data']['max_id']
    comments = jsonobj['data']['data']
    for comment in comments:
        text = comment['text']
    if(nextid > 0):
        getWeibo(weiboId,makeUrl(weiboId,nextid)) 


def makeUrl(weiboId,maxId):
    url = 'https://m.weibo.cn/comments/hotflow?id=' + weiboId + '&mid=' + weiboId
    if(maxId == 0):
        return None
    if(maxId > 0):
        url += '&max_id=' + str(maxId)
    url += '&max_id_type=0'
    return url


def main():
    weiboId = dealURL()
    firstRequestURL = makeUrl(weiboId,-1)
    getWeibo(weiboId,firstRequestURL)


def dealURL():
    if(len(sys.argv) == 2):
        firstUrl = sys.argv[1]
    else:
        firstUrl = 'https://m.weibo.cn/detail/4561046402236526'
    pattern = re.compile(r'detail/[0-9]+')
    WeiboId = re.search(pattern,firstUrl)
    if WeiboId:
        return WeiboId.group(0)[7:]
    return None

url = "https://img-ys01.didistatic.com/static/gaia_datafile/20201012150828attr.txt?expire=1604833074&signiture=HsFtbI6--67AadwNbeCNc6hY-kcvBtFyeaTTsK7AHLc="
response = requests.get(url=url, headers=headers, verify=False)
print("waiting")
f= open("didi.txt","w+")
f.write(response.text)
f.close()