import os
import shutil
import numpy as np
import math

np.random.seed(123)

class DirectoryAssistor():
    def __init__(self):
        pass

    def make_directory(self, path, name):
        """
        Makes a new folder

        Params
        path: str where you want to create a folder
        name: name of folder
        """
        dir = os.path.join(path, name)
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    def rename_directory(self, path, name, new_name):
        """
        Renames folder

        Params
        path: str where you the folder is located
        name: str old name of folder
        new_name: str new name of folder
        """
        oldpath = os.path.join(path, name)
        newpath = os.path.join(path, new_name)
        if os.path.exists(oldpath):
            os.rename(oldpath, newpath)
   
    def move_contents(self, filepath, new_folder):
        shutl.move(self, filepath, new_folder)

    def delete_files(self, filepath):
        os.remove(filepath)

    def traverse(self, path):
        """
        for each folder in the specified path, 
        prints out the folder directory and the number of the files in the folder

        Params
        path: str where you the folder is located
        """
        for root, dirs, files in os.walk(path):
            print(f'{dirs} : {len(files)}')
    
    def create_content_list(self, path):
        """
        Creates a list of all items in the specified path

        Params
        path: str where you the folder is located
        """
        return [name for name in os.listdir(path)]

    def copy_files(self, old_path, new_path, new_folder):
        """
        copies files from one folder to a new folder

        Params
        old_path: str where the folder being copied is located
        new_path: str where the folder being copied to will be located
        new_folder: str name of new folder
        """
        file_lst = self.create_content_list(old_path)
        for file in file_lst:
            shutil.copy(old_path+file, new_path+new_folder)

    def copy_all_files(self, old_path, new_path, new_folder):
        """
        copies all files from the folders in old_path to a new_folder located
        in new_path

        Params
        old_path: str where the folder being copied is located
        new_path: str where the folder being copied to will be located
        new_folder: str name of new folder
        """
        folder_lst = self.create_content_list(old_path)
        for folder in folder_lst:
            file_lst = self.create_content_list(old_path+folder)
            folder_path = old_path+folder+'/'
            for file in file_lst:
                shutil.copy(folder_path+file, new_path+new_folder)

    def train_test_split(self, old_path, old_folder, new_path, new_folder, split=0.75,
                         amount=None):
        """
        Creates a train and test split by randomly selecting files from the 
        specified old_folder and copies them to a new folder in the new_path

        Params
        old_path: str where the folder being copied is located
        old_folder: str folder which has the files to be copied
        new_path: str where the folder being copied to will be located
        new_folder: str name of new folder
        split: float specifies the percantage of files in the folder to be allocated
        to train with the rest are used for testing
        """
        file_lst = self.create_content_list(old_path+old_folder)
        
        if amount == None:
            cropped_lst = file_lst
        else:
            cropped_lst = np.random.choice(file_lst, amount, replace=False)
        
        file_number = len(cropped_lst)
        file_number_train = round(file_number*split)
        print(f'Train:{file_number_train}')
        train_files = np.random.choice(cropped_lst, file_number_train, replace=False)
        test_files = list(cropped_lst.copy())
        
        for file in train_files:
            test_files.remove(file)
        
        print(f'Test: {len(test_files)}')
        self.make_directory(new_path, 'TrainDocs')
        self.make_directory(new_path, 'TestDocs')
        train_path = new_path+'TrainDocs'
        test_path = new_path+'TestDocs'
        self.make_directory(train_path+'/', new_folder)
        self.make_directory(test_path+'/', new_folder)
        for file in train_files:
            shutil.copy(old_path+old_folder+'/'+file, new_path+'TrainDocs'+'/'+
                        new_folder)
        for file in test_files:
            shutil.copy(old_path+old_folder+'/'+file, new_path+'TestDocs'+'/'+
                        new_folder)
        return train_files, test_files

    def train_test_all_folders(self, old_path, new_path, split=0.75, 
                               amount=None):
        """
        Creates a train and test split from all the folders in old_path directory
        and puts them into a train and test folder located in new_path

        Params
        old_path: str where the folders being copied is located
        new_path: str where the folder being copied to will be located
        split: float specifies the percantage of files in the folder to be allocated
        to train with the rest are used for testing
        """
        folder_list = self.create_content_list(old_path)
        for folder in folder_list:
            self.train_test_split(old_path, folder, new_path, folder, 
                                  split=split, amount=amount)

    def train_test_holdout(self, old_path, new_path, train_test_folder,
                           holdout_split=0.75 , split=0.75, initial_amt=None,
                           second_amt=None):
        folder_list = self.create_content_list(old_path)
        for folder in folder_list:
            self.train_test_split(old_path, folder, new_path, 
                                  folder, split=split, amount=initial_amt)
        for folder in folder_list:
            self.train_test_split(new_path+'TrainDocs/', folder, 
                                  train_test_folder, folder, split=split,
                                  amount=second_amt)


        
execute = True
if __name__ == "__main__" and execute:
    
    # Train Test Holdout Test on txt files
    
    old_path = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
EC2Data/txtFiles/'
    new_path = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/TrainTestHoldout/'
    train_test_folder = '/Users/justinlansdale/Documents/Galvanize/\
Capstone2/contractTxts/TrainTestHoldout/TrainTest/'

    da = DirectoryAssistor()
    da.train_test_holdout(old_path, new_path, train_test_folder, initial_amt=600)


    
    