from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
from directoryassistor import DirectoryAssistor
import time
from directoryassistor import DirectoryAssistor

class WebScraper():
    def __init__(self, domain, download_dir):
        self.domain = domain
        self.r = requests.get(domain)
        self.src = self.r.content
        self.soup = bs(self.src, 'lxml')
        self.download_dir = download_dir
    
    def status(self):
        print(self.r.status_code)
        
    def get_urls(self, primary_tag, url_tag, tag2, url_attr='href', url_tag_loc=0, tag2_loc=0):
        count = 1
        self.url_dic = {}
        self.tags = self.soup.find_all(primary_tag)
        for tag in self.tags:
            url = tag.find(url_tag)
            t2 = tag.find(tag2)
            if t2 != None and count>tag2_loc:
                self.url_dic[t2.text] = []
                urls = []
                t2_txt = t2.text
            if url != None and count > url_tag_loc:
                self.url_dic[t2_txt].append(url.attrs[url_attr])
            count += 1
    
    def get_docs(self):
        ds = DirectoryAssistor()
        for key, values in self.url_dic.items():
            new_key = key.replace(" ","")
            ds.make_directory(self.download_dir, new_key)
            new_dir = self.download_dir+key
            strip_dir = new_dir.replace(" ","")
            options = webdriver.ChromeOptions()
            prefs = {'download.default_directory': strip_dir}
            options.add_experimental_option('prefs', prefs)
            print(strip_dir)
            for val in values:
                driver = webdriver.Chrome(chrome_options=options)
                driver.get(domain+val)
                time.sleep(5)
                button = driver.find_element_by_tag_name('button').click() 
                time.sleep(10)
                driver.quit()







execute = True
if __name__ == "__main__" and execute:

    domain = 'https://www.printablecontracts.com/'
    dir1 = '/Users/justinlansdale/temp/testdir/'
    ws = WebScraper(domain, dir1)
    ws.get_urls('li', 'a', 'b', url_tag_loc=29, tag2_loc=28)
    ws.get_docs()

#     domain2 = 'https://www.hloom.com/resources/templates/more/contract'
#     dir2 = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
# ContractReader/data2'
#     ws2 = WebScraper(domain, dir1)
#     ws2.status()
#     ws2.get_urls('div', 'a', 'h2', url_tag_loc=1, tag2_loc=1)
#     print(len(ws2.tags))

    # r = requests.get(domain)
    # print(r.status_code)

    # src = r.content

    # soup = bs(src, 'lxml')


    # remove = 29
    # count = 1
    # # # for li_tag in soup.find_all('li'):
    # # #     a_tag = li_tag.find('a')
    # # #     b_tag = li_tag.find('b')
    # # #     if a_tag != None and count > 29:
    # # #         urls.append(a_tag.attrs['href'])
    # # #     if b_tag != None:
    # # #         contractNames.append(b_tag.text)
    # # #     count += 1
    # contractDict = {}

    # for li_tag in soup.find_all('li'):
    #     a_tag = li_tag.find('a')
    #     b_tag = li_tag.find('b')
    #     if b_tag != None and count>28:
    #         contractDict[b_tag.text] = []
    #         urls = []
    #         bTag = b_tag.text
    #     if a_tag != None and count > 29:
    #         contractDict[bTag].append(a_tag.attrs['href'])
    #     count += 1


    # newDic = {k: v for k, v in contractDict.items() if k.startswith('Child Care')}
    # print(newDic)

    # ds = DirectoryAssistor()
    # dir = '/Users/justinlansdale/Documents\
    # /Galvanize/Capstone2/ContractReader/data/'

    # for key, values in contractDict.items():
    #     new_key = key.replace(" ","")
    #     ds.make_directory(dir, new_key)
    #     new_dir = dir+key
    #     strip_dir = new_dir.replace(" ","")
    #     options = webdriver.ChromeOptions()
    #     prefs = {'download.default_directory': strip_dir}
    #     options.add_experimental_option('prefs', prefs)
    #     print(strip_dir)
    #     for val in values:
    #         driver = webdriver.Chrome(chrome_options=options)
    #         driver.get(domain+val)
    #         time.sleep(5)
    #         button = driver.find_element_by_tag_name('button').click() 
    #         time.sleep(10)
    #         driver.quit()




    # # dir = '/Users/justinlansdale/Documents\
    # # /Galvanize/Capstone2/ContractReader/data/testdirs'
    # # options = webdriver.ChromeOptions()
    # # prefs = {'download.default_directory': dir}
    # # options.add_experimental_option('prefs', prefs)
    # # driver = webdriver.Chrome(chrome_options=options)

    # # driver.get(domain+'Express_Warranty.php')
    # # time.sleep(5)
    # # button = driver.find_element_by_tag_name('button').click()
    # # time.sleep(10)
    # # driver.quit()

