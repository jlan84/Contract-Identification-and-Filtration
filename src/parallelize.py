import pytesseract
import cv2
import re
import os
import glob
import concurrent.futures
import time
from pdf2image import convert_from_path
from directoryassistor import DirectoryAssistor

def convert_to_img(directory, file_name, file_type='.pdf', dpi=500, 
                       verbose=True):
        """
        Converts each page in a file to an image

        Params:
        directory: str of the directory where the files are located must end in 
        '/'
        file_name: str name of the file
        file_type:str default '.pdf'
        dpi: int dots per inch
        verbose: bool default is True
        """

        if verbose:
            print(f'Processing {file_name}')
               
        pages = convert_from_path(directory+file_name, dpi=dpi)
        image_counter = 1
        image_names = []

        if verbose:
            print('Creating images')       
        
        for page in pages:
            print(f'page{image_counter}')
            image_name = 'page_'+str(image_counter)+'.jpg'
            image_names.append(image_name)
            page.save(image_name, 'JPEG')
            image_counter += 1
        print(f'Total Pages to Process {image_counter}')
        return image_names
        
def ocr(img_name):
    """
    Takes an image in and returns a .txt file

    Params:
    img_name: str the name of the image to be read in
    """

    out_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/Arch_Engineer'
    img = cv2.imread(img_name)
    text = pytesseract.image_to_string(img,lang='eng',config='--psm 6')
    out_file = re.sub(".jpg",".txt",img_name)
    out_path = out_dir + out_file
    fd = open(out_path,"w")
    fd.write("%s" %text)
    return out_file

def merge_pages(page_lst, out_dir, file_name):
    """
    Merges pages from the document into one .txt file

    Params:
    page_lst: lst of pages in the directory
    out_dir: str dir where the pages are stored
    file_name: str name of file
    """

    main_txt = ''
    for page in page_lst:
        with open(out_dir+page, 'r') as f:
            data = f.read()
        main_txt += data+'\n'
    txt_name = file_name.replace('.pdf','.txt')
    with open(out_dir+txt_name, 'w') as f:
        f.write(main_txt)
    

def main(directory, file_name,out_dir, file_type='.pdf', dpi=500, verbose=True):
    """
    Walks through all the pages in a file, converts to image, and uses OCR to 
    convert image to .txt files.  It then deletes the image files and merges the
    individual page .txt files.  After it deletes the individual page .txt files

    Params:
    directory: str where the files are stored
    file_name: str name of file
    out_dir: str the directory where the text files are going
    file_type:str default '.pdf'
    dpi: int dots per inch
    verbose: bool default is True
    """

    da = DirectoryAssistor()
    out_dir = out_dir
    image_list = convert_to_img(directory=directory, file_name=file_name)
    page_lst = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        for img_path,out_file in zip(image_list,executor.map(ocr,image_list)):
            print(img_path.split("\\")[-1],',',out_file,', processed')
            page_lst.append(out_file)
    
    for img in image_list:
            da.delete_files(img)

    merge_pages(page_lst=page_lst, out_dir=out_dir, file_name=file_name)
    
    for page in page_lst:
        da.delete_files(out_dir+page) 

 
if __name__ == '__main__':
    directory='/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractPdfs/Arch_Engineer/'
    out_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone2/\
contractTxts/Arch_Engineer'
    da = DirectoryAssistor()
    doc_lst = da.create_content_list(directory)
    start = time.time()
    
    for i in range(len(doc_lst)):
        file_name = doc_lst[i]
        try:
            main(directory, file_name, out_dir)
        except:
            continue
    end = time.time()
    print(end-start)