import os
import glob
import pandas as pd
import numpy as np
import re
from openpyxl import load_workbook

def year_month(s):
    # find all 4-digits occurrences, chooses the last one and the 3 following characters
    pattern = re.compile(r'(\d{4})(.{3})')
    matches = pattern.findall(s)

    if matches:
        dernier_match = matches[-1]
        year = dernier_match[0]
        month = dernier_match[1][-2:]
        return year, month
    else:
        return None

root_directory = "*************************************************************"

#glob.glob finds all files corresponding to a defined file type
excel_files = glob.glob(os.path.join(root_directory, "**\\*.xls*"), recursive=True)

#creates a list that contains the paths of the Excel files beginning with "2_e_Pays prov_dernier mois" and containing 7 or 8 backslashes
filtered_files = [file for file in excel_files if os.path.basename(file).startswith("2_e_Pays prov_dernier mois") and (file.count("\\")==7 or file.count("\\")==8)]

result = pd.DataFrame()  

#go through every file in the list
for file in filtered_files:
    try:
        #loads the Excel file
        xls = pd.ExcelFile(file)

        #takes the names of the sheets
        regions = xls.sheet_names

        #go through every sheet
        for region in regions:
            #reads the content of the sheet into a dataframe
            df = pd.read_excel(file, sheet_name=region,header=None)
            df = pd.read_excel(file,sheet_name=region,header=None)
            df = df.iloc[:, [0, 4, 5]]
            df = df.iloc[9:-1, :]
            df['year.month'] = "01."+year_month(file)[1]+"."+year_month(file)[0]
            df['region']=region
            result=pd.concat([result,df])
            print(file)

    except Exception as e:
        print(f"Erreur lors du traitement de {file}: {e}")

result = result.rename(columns={0: 'Pays', 4: 'Arrivées', 5: 'Nuitées', 'year.month': 'Année.mois', 'region': 'Région'})

os.chdir("C:\\Users\\Chapatte\\Documents\\Python et R pour HESTA")

result.to_csv('Données aggrégées.csv', sep=';', index=False, encoding='latin-1')