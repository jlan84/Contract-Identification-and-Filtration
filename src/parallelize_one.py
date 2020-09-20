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
    out_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone3/EC2Data/c\
hangeFolder/Amended/'

    img = cv2.imread(img_name)
    text = pytesseract.image_to_string(img,lang='eng',config='--psm 6')
    out_file = re.sub(".jpg",".txt",img_name)
    out_path = out_dir + out_file
    fd = open(out_path,"w")
    fd.write("%s" %text)
    return out_file

def merge_pages(page_lst, out_dir, file_name):
    main_txt = ''
    for page in page_lst:
        with open(out_dir+page, 'r') as f:
            data = f.read()
        main_txt += data+'\n'
    txt_name = file_name.replace('.pdf','.txt')
    with open(out_dir+txt_name, 'w') as f:
        f.write(main_txt)
    

def main(directory, file_name,out_dir, file_type='.pdf', dpi=500, verbose=True):
    da = DirectoryAssistor()
    out_dir = out_dir
    image_list = convert_to_img(directory=directory, file_name=file_name)
    page_lst = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for img_path,out_file in zip(image_list,executor.map(ocr,image_list)):
            print(img_path.split("\\")[-1],',',out_file,', processed')
            page_lst.append(out_file)
    
    for img in image_list:
            da.delete_files(img)

    merge_pages(page_lst=page_lst, out_dir=out_dir, file_name=file_name)
    
    for page in page_lst:
        da.delete_files(out_dir+page)



 
if __name__ == '__main__':
    directory='/Users/justinlansdale/Documents/Galvanize/Capstone3/EC2Data/\
changeFolder/pdfs/Original/'
    out_dir = '/Users/justinlansdale/Documents/Galvanize/Capstone3/EC2Data/\
changeFolder/Amended/'

    da = DirectoryAssistor()
    doc_lst = da.create_content_list(directory)
    print(doc_lst)
    start = time.time()
    
    for i in range(len(doc_lst)):
        file_name = doc_lst[i]
        try:
            main(directory, file_name, out_dir)
        except:
            continue
    end = time.time()
    print(end-start)