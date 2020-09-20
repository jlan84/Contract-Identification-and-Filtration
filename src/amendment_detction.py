from parallelize_one import *
import numpy as np
from directoryassistor import DirectoryAssistor



class AmendmentDetector():

    def __init__(self, original_doc_dir, amended_doc_dir, original_txt_dir,
                 amended_txt_dir):
    
        self.original_doc_dir = original_doc_dir
        self.amended_doc_dir = amended_doc_dir
        self.original_txt_dir = original_txt_dir
        self.amended_txt_dir = amended_txt_dir
        self.da = DirectoryAssistor()

    def convert_original_to_txt(self):
        doc_lst = self.da.create_content_list(self.original_doc_dir)

        for i in range(len(doc_lst)):
            file_name = doc_lst[i]
            try:
                main(self.original_doc_dir, file_name, self.original_txt_dir)
            except:
                continue
    
    def convert_amended_to_txt(self):
        doc_lst = self.da.create_content_list(self.amended_doc_dir)

        for i in range(len(doc_lst)):
            file_name = doc_lst[i]
            try:
                main(self.amended_doc_dir, file_name, self.amended_txt_dir)
            except:
                continue
    
    def read_in_files(self):
        original_lst = self.da.create_content_list(self.original_txt_dir)
        amended_lst = self.da.create_content_list(self.amended_txt_dir)
        for doc in original_lst:
            try:
                with open(self.original_txt_dir+doc, 'r') as f:
                   self.original = f.read().replace('\n',' ')
            except:
                continue
        
        for doc in amended_lst:
            try:
                with open(self.amended_txt_dir+doc, 'r') as f:
                    self.amended = f.read().replace('\n',' ')
            except:
                continue
    
    def print_changes(self):
        original_lst = self.original.split()
        amended_lst = self.amended.split()
        original_value = []
        amended_value = []
        change_ref = []
        for i in range(len(original_lst)):
            if original_lst[i] != amended_lst[i]:
                original_value.append(original_lst[i])
                change_ref.append(list(original_lst[i-40:i+40]))
                amended_value.append(amended_lst[i])
        
        for i in range(len(original_value)):
            print(f'\n Change # {i+1}: {original_value[i]} changed to \
{amended_value[i]} \n \n Reference text: {" ".join(change_ref[i])} \n')


        


    



if __name__ == '__main__':
    original_doc_dir='/Users/justinlansdale/Documents/Galvanize/Capstone3/EC2Data/\
changeFolder/pdfs/Original/'
    original_txt_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone3/EC2Data/\
changeFolder/Original/'

    amended_doc_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone3/EC2Data/\
changeFolder/pdfs/Amended/'
    amended_txt_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone3/EC2Data/\
changeFolder/Amended/'

    ad = AmendmentDetector(original_doc_dir, amended_doc_dir, original_txt_dir,
                           amended_txt_dir)
    
    ad.read_in_files()
    ad.print_changes()
    
    # ad.convert_original_to_txt()