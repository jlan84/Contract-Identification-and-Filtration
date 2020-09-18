import pandas as pd
from EDA import make_sns_bar_plot
import seaborn as sns
import matplotlib.pyplot as plt
from selenium import webdriver
from bs4 import BeautifulSoup
from directoryassistor import DirectoryAssistor
import time
import logging


plt.style.use('fivethirtyeight')

class DocumentExtractor():

    def __init__(self, df):
        self.df = df

    def replace_col_values(self, lst, replacement_lst):
        '''
        Replaces the lst values in self.df with the replace_value

        Params
        lst: list of lists that correspond to the replacement values

        replacement_lst: lst of strs with which to replace the list values 
        in self.df
        
        '''
        for i in range (len(lst)):
            self.df.replace(lst[i], replacement_lst[i], inplace = True)    
    
    def process_df(self, columns_to_keep):
        '''
        Returns a new df with the columns from the columns_to_keep list and all
        null values removed

        Params
        df: DataFrame that needs to be processed
        columns_to_keep: list with the columns to keep
        '''

        self.df = self.df.dropna(subset=['Contract PDF', 'Contract Type']).copy()
        column_lst = list(self.df.columns)
        columns_to_drop = column_lst.copy()
        for col in columns_to_keep:
            columns_to_drop.remove(col)
        self.df.drop(columns=columns_to_drop, inplace=True)
        return self.df.reset_index()

    def subset(self, contract_type, subset=2000, random_state=123):
        new_df = (self.df[self.df['Contract Type'] == contract_type].
                  sample(n=subset, replace=False, random_state=random_state))
        contract_dic = {contract_type: list(new_df['Contract PDF'])}
        return new_df, contract_dic 
    
    def scrape_pdf(self, directory, contract_dic):
        ds = DirectoryAssistor()
        count = 0
        for key, values in contract_dic.items():
            ds.make_directory(directory, key)
            contract_dir = directory+key
            options = webdriver.ChromeOptions()
            prefs = {"download.default_directory": contract_dir,
                    "plugins.always_open_pdf_externally": True}
            options.add_experimental_option('prefs', prefs)
            for val in values:
                driver = webdriver.Chrome(options=options)
                driver.get(val)
                time.sleep(5)
                elm = driver.find_element_by_xpath("//*[@id='pdfFrame']")
                url = elm.get_attribute('src')
                driver.get(url)
                time.sleep(7)
                driver.quit()
                ds.rename_directory(contract_dir, 'DPSWebDocumentViewer.pdf',f'document\
{count}.pdf')
                count += 1



if __name__ == "__main__":

    df = pd.read_csv('../data/CityofChicago/Contracts.csv', low_memory=False)

    de = DocumentExtractor(df)
    
    delegate_agency = ['DELEGATE AGENCY']

    comptroller = ['COMPTROLLER-OTHER ']

    arch_engin = ['ARCH/ENGINEERING']

    pro_service = ['PRO SERV CONSULTING $250,000orABOVE', 'PRO SERV-AVIATION',
                   'PRO SERV CONSULTING UNDER $250,000','PRO SERV-SMALL ORDERS',
                   'Professional Services', 'PRO SERV', 
                   'PRO SERV-BUSINESS CONSULTING', 'PRO SERV-SOFTWARE/HARDWARE',
                   'CONSTRUCTION SERVICES']
    
    construction = ['CONSTRUCTION-LARGE $3MILLIONorABOVE',
                    'CONSTRUCTION-AVIATION', 'CONSTRUCTION-GENERAL',
                    'CONSTRUCTION', 'Construction', 'ROOFING']                
    
    commodities = ['COMMODITIES', 'COMMODITIES-SMALL ORDERS',
                   'COMMODITIES-AVIATION']
    
    contract_type_lst = ['Delegate_Agency','Construction','Comptroller',
                         'Commodities', 'Arch_Engineer', 'Prof_Services']
    contract_val_lst = [delegate_agency, construction, comptroller,
                        commodities, arch_engin, pro_service]
    
    de.replace_col_values(contract_val_lst, contract_type_lst)

    # for i in range(len(contract_type_lst)):
    #     de.replace_col_values(contract_val_lst[i], contract_type_lst[i])
    # # df.replace(pro_service, 'Prof_Services', inplace=True)
    de.process_df(['Contract PDF', 'Contract Type'])
    
    new_df, contract_dic = de.subset('Delegate_Agency', subset=5)
    
    directory = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
Contract-Classifier/data/CityofChicago/ChicagoContracts/'
    de.scrape_pdf(directory, contract_dic)
    # prefs = {"plugins.always_open_pdf_externally": True
    #         }
    
    # options = webdriver.ChromeOptions()
    
    # options.add_experimental_option('prefs', prefs)
    # # logging.basicConfig(level=logging.DEBUG)
    # driver = webdriver.Chrome(chrome_options=options)
    # driver.get('https://www.troweprice.com/content/dam/trowecorp/Pdfs/TRPIL%20MiFID%20II%20Execution%20Quality%20Report%202017.pdf')

    # contract_type_group = (de.df.groupby('Contract Type').count().
    # sort_values('Contract PDF', ascending=False))
    # contract_type_group.reset_index(inplace=True)
    # # delegate_grp = de.df[de.df['Contract Type'] == 'DELEGATE AGENCY']
    # comp_trol_grp = de.df[de.df['Contract Type'] == 'COMPTROLLER-OTHER']
    # print(comp_trol_grp['Contract PDF'].iloc[0])
    # driver.page_source
    
    

    
