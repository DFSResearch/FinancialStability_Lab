# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 19:59:43 2018

@author: FS_ARP_2
"""

import urllib3
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pickle
import random

def sub_dict_strict(somedict, somekeys):
    return dict([ (k, somedict[k]) for k in somekeys ])

def sub_dict_remove_strict(somedict, somekeys):
    return dict([ (k, somedict.pop(k)) for k in somekeys ])

class kase_handler:
    
    
    def __init__(self):
        self.base_url = 'http://kase.kz/ru/'
        self.save_path = 'C:\\Users\\FS_ARP_2.CORP\\Desktop\\kase\\'
        self.http_handler = urllib3.PoolManager()
        self.gov = 'gsecs/show/'
        self.last_bond = ''
        self.__bonds_proceeded = 0
        
    def __cook_soup__(self, url):
        
        r = self.http_handler.request('GET', url)
        i=1
        while r.status!=200:
            if i%10 == 0:
                url = url.replace('_','.')
            if i%20 == 0:
                break
            r = self.http_handler.request('GET', url)
            print(r.status, url)
            time.sleep(random.randint(2,5))
            i+=1
        if r.status == 200:
            soup = BeautifulSoup(r.data, "lxml")
        else:
            soup = None
        
        r.close()
        
        return soup
    
    def dump_db(self, data, file='data.pkl'):
        
        fpath = self.save_path + file
        f = open(fpath, 'wb')
        pickle.dump(data, f)
        f.close()
        
        print(f'Data saved at {self.save_path} as {file}')
    
    def load_db(self, file='data.pkl'):
        
        fpath = self.save_path + file
        f = open(fpath, 'rb')
        data = pickle.load(f)
        f.close()
        
        print(f'Data from {fpath} loaded')
        return data
    
    def get_by_code(self, sector = 'gov', code = 'mum120_0003'):
        
        if sector == 'gov':
            
            url = self.base_url + self.gov + code.upper()
            soup = self.__cook_soup__(url)
            info = soup.select(f'#characteristic > div.accordion__body > div.info-table')[0]
            info = info.text.split('\n\n\n')
            result = dict()
            for i in info:
                tmp = i.split(':')
                tmp[0] = tmp[0].strip()
                tmp[1] = tmp[1].strip()
                result[tmp[0]] = tmp[1]
                #print(tmp)
        
        return result

    def get_coupon_schedule(self, bond_dict, display = False):
        
        schedules = {}
        if isinstance(bond_dict, str):
            iterations = [bond_dict]
        else:
            iterations = bond_dict.keys()
        
        for bond in iterations:
            
            url = self.base_url + self.gov + bond
            soup = self.__cook_soup__(url)
            
            if soup != None:
                soup = soup.find_all('div', {'class':'modal fade', 'id':'currencyChartModal'})[0]
                soup = soup.find_all('table')[0]
                
                row_num = 0
                max_row = len(soup.find_all('tr'))-1
                schedule = pd.DataFrame(columns = ['Start_date', 'Rate', 'Fixation_date', 'End_date'], index = range(max_row))
                
                for row in soup.find_all('tr')[1:]:
                    col_num = 0
                    columns = row.find_all('td')[1:]
                    
                    for column in columns:
                        schedule.iat[row_num, col_num] = column.get_text().strip()
                        #print(column.get_text().strip())
                        col_num += 1
                    row_num += 1
            else:
                    schedule = None
                    print(f'None for {bond}')
            
        schedules[bond] = schedule
        
        self.last_bond = bond
        self.__bonds_proceeded+=1
        
        if display and self.__bonds_proceeded%200 == 0:
            print(self.__bonds_proceeded)
        
        return schedules
        
ks = kase_handler()

data = ks.load_db()

udata = {}
for i in data.keys():
    
    udata[i] = ks.get_coupon_schedule(i, display=True)
    
ks.dump_db(udata, file = 'schedule.pkl')

for i in udata.keys():
    
    data[i]['График купонных выплат'] = udata[i][i]
    

ks.dump_db(data)

c_df = pd.DataFrame({'symb':[], 'rate':[]})
for i in data.keys():
    try:
        c_df = c_df.append({'symb':i, 'rate':data[i]['График купонных выплат'].Rate[0]}, ignore_index = True)
    
    except:
        print(i)

data = dict()

codes = list(df.symbol.unique())

for i, code in tqdm(enumerate(codes)):
    bond = ks.get_by_code(code=code)
    if i%100 == 0:
        print(f'{i} bonds done!!')
    data[bond.get('Код бумаги')] = bond
    print(bond.get('Код бумаги'))

ks.dump_db(data)

for i, row  in c_df.iterrows():
    
    c_df.loc[i, 'kase'] = clean_data.query(f'symbol=="{row.symb}"').coupon_rate.mean()
    
c_df.rate = c_df.rate.str.replace(',','.')
c_df.loc[303,'rate'] = None
c_df.rate = pd.to_numeric(c_df.rate)/100
