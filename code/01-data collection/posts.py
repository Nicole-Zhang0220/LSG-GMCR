import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
import json

# set cookies
cookies = {
    'SINAGLOBAL': '568365326591.7821.1702080893431',
    'SCF': 'ApDYB6ZQHU_wHU8ItPHSso29Xu0ZRSkOOiFTBeXETNm7k7YlpnahLGVhB90-mk0xFNznyCVsjyu9-7-Hk0jRULM.',
    'SUB': '_2A25Id8HLDeRhGeNN41ES9CzNzDSIHXVrDVsDrDV8PUNbmtANLXDNkW9NSYW6fC4KGVbVnCuommwHhHoTAJMiMiob',
    'SUBP': '0033WrSXqPxfM725Ws9jqgMF55529P9D9WW8S_g7r4Oo7uo2nVW7ZgD75JpX5KzhUgL.Fo-01he0ShzpS0n2dJLoIN-LxK-LB-qLBo.LxKMLB-eL1K2LxKML1-2L1hBLxK-LBK-LB.BLxKML1-eL12zLxKML1-2L1hBLxKBLB.2LB.2LxKqLBozLBK2LxKqL1-eL1h.LxKqL122LBo-LxKqL122L1KMLxK-L122L1-zLxK.L1KnLB.qt',
    'ALF': '1733616922',
    '_s_tentry': 'weibo.com',
    'Apache': '568365326591.7821.1702080893431',
    'ULV': '1702080893446:1:1:1:568365326591.7821.1702080893431:',
}


def get_the_list_response(q='话题', n='1', p='页码'):
    headers = {
        'authority': 's.weibo.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'referer': 'https://s.weibo.com/weibo?q=%23%E6%96%B0%E9%97%BB%E5%AD%A6%E6%95%99%E6%8E%88%E6%80%92%E6%80%BC%E5%BC%A0%E9%9B%AA%E5%B3%B0%23&nodup=1',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69',
    }

    params = {
        'q': q,
        'nodup': n,
        'page': p,
    }
    response = requests.get('https://s.weibo.com/weibo', params=params, cookies=cookies, headers=headers)
    return response


def parse_the_list(text):
    soup = BeautifulSoup(text)
    divs = soup.select('div[action-type="feed_list_item"]')
    lst = []
    for div in divs:
        mid = div.get('mid')
        uid = div.select('div.card-feed > div.avator > a')
        if uid:
            uid = uid[0].get('href').replace('.com/', '?').split('?')[1]
        else:
            uid = None
        time = div.select('div.card-feed > div.content > div.from > a:first-of-type')
        if time:
            time = time[0].string.strip()
        else:
            time = None
        p = div.select('div.card-feed > div.content > p:last-of-type')
        if p:
            p = p[0].strings
            content = '\n'.join([para.replace('\u200b', '').strip() for para in list(p)]).strip()
        else:
            content = None
        star = div.select('ul > li > a > button > span.woo-like-count')
        if star:
            star = list(star[0].strings)[0]
        else:
            star = None
        lst.append((mid, uid, content, star, time))
    df = pd.DataFrame(lst, columns=['mid', 'uid', 'content', 'star', 'time'])
    return df


def get_the_list(q, p):
    df_list = []
    for i in range(1, p + 1):
        response = get_the_list_response(q=q, p=i)
        if response.status_code == 200:
            df = parse_the_list(response.text)
            df_list.append(df)
            print(f'第{i}页解析成功！', flush=True)

    return df_list


if __name__ == '__main__':
    q = '#东三省用电高峰拉闸限电#'
    p = 31
    df_list = get_the_list(q, p)
    df = pd.concat(df_list)
    df.to_csv(f'{q}.csv', index=False)
