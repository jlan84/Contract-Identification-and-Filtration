<p align="center">
<img src="images/coffee_and_contracts.png"  height="100" width="500" />
</p>


</br>
</br>
</br>
</br>


# Contract Identification and Filtration

## Introduction

For normal everyday people, reading contracts is not the most appealing endeavor and we deal with them on a regular basis.  Whether it's agreeing to a new update on your smart phone or signing all of the documents when closing on a new home, few of us actually read through the contract to ensure we are not signing our lives away.  The ultimate goal of this project is to distinguish the important parts of a contract from the rest.

## Process

<p align="center">
<img src="images/work_flow.png"  height="100" width="500" />
</p>

### Creating the Database

Contracts are typically private documents and therefore obtaining enough contracts to genereate a model was no easy task.  I obtained all of my contracts for this model from the City of Chicago portal (https://data.cityofchicago.org).  Below is the workflow used to obtain these documents

<p align="center">
<img src="images/scrape_workflow.png"  height="400" width="700" />
</p>


### Processing the Contracts

These contracts came in as mostly scanned pdfs, so in order to process them I used Optical Character Recognition (OCR).  OCR is computationally expensive so in order to optimize my time, I used cloud computing on AWS.  I split the 3600 pdfs I was going to use for the model evenly between six m5a.8xlarge EC2 instances.  Each instance generated a .jpg image file for each page of the .pdf files. I then split these .jpg images onto the 32 cores and used OCR to process the images individually in order to reduce processing time.  These returned individual .txt files for each image, which I combined to create one .txt file for the original .pdf.  The process flow for this is displayed in the image below.

<p align="center">
<img src="images/pdf_process_workflow.png"  height="400" width="700" />
</p>

## The Contracts

Contracts are typically private documents and therefore obtaining enough contracts to genereate a model was no easy task.  