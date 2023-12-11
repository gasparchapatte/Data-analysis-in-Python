from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup as bs
import pandas as pd
from scrapy.selector import Selector
import time
import re

hotel_no=94

def scrape_page(i,driver):
    data = {'Code postal': [],'Ville': [],'Etoiles': [], 'Contenu positif': [], 'Contenu négatif': [], 'Langue': [], 'Type': [], 'Nuits': [], 'Date': [], 'Pays': [], 'Occurrences': [], 'Hôtel no': []}


    soup = bs(driver.page_source, 'html.parser')

    #Get the number of stars if any
    if "Ce nombre d'étoiles" in driver.page_source:
        # Search the spans with the specified classes "fcd9eec8fb d31eda6efc c25361c37f"
        spans_with_class = driver.find_elements(By.CSS_SELECTOR, 'span.fcd9eec8fb.d31eda6efc.c25361c37f')
        etoiles=str(len(spans_with_class))+' étoiles'
        print (str(len(spans_with_class))+' étoiles')
    else:
        etoiles="Pas d'étoiles"
        print("Pas d'étoiles sur cette page.")


    code_postal=''
    ville=''
    span = soup.find('span', class_='hp_address_subtitle')

    if span:
        contenu_span = span.text.strip()

        # Find the post code (4 following up numbers)
        match = re.search(r'\b(\d{4})\b', contenu_span)

        if match:
            code_postal = match.group(0)
            ville_pays=contenu_span.split(code_postal)[1][1:]
            ville=ville_pays.split(',')[0]

        else:
            print("Aucun code postal trouvé.")

    else:
        print("Balise <span> avec la classe 'hp_address_subtitle' non trouvée.")



    

    review_blocks = soup.find_all('div', class_='c-review-block')

    for review_block in review_blocks:
        if (len(review_block.find_all('div', class_='bui-list__body'))==3):
            bui_list_bodies = review_block.find_all('div', class_='bui-list__body')[2].get_text().replace("\n", "")
            nuits = review_block.find_all('div', class_='bui-list__body')[1].get_text().replace("\n", "").split(' ')[0]
            date = review_block.find_all('div', class_='bui-list__body')[1].get_text().replace("\n", "").split(' · ')[1]
        else:
            bui_list_bodies = review_block.find_all('div', class_='bui-list__body')[1].get_text().replace("\n", "")
            nuits = review_block.find_all('div', class_='bui-list__body')[0].get_text().replace("\n", "").split(' ')[0]
            date = review_block.find_all('div', class_='bui-list__body')[0].get_text().replace("\n", "").split(' · ')[1]

        pays= review_block.find_all('span')[1].text.strip()

        reviews = review_block.find_all('div', class_='c-review')

        for review in reviews:
            contenu_positif=''
            contenu_negatif=''
            lang_attribut=''
            review_rows = review.find_all('div', class_='c-review__row')
            for review_row in review_rows:
                p_tags = review_row.find_all('p')

                for p_tag in p_tags:
                    span_tags = p_tag.find_all('span', class_='c-review__body')
                    lang_attribut = span_tags[0].get('lang', '')
                    sentiment = ""
                    prefix_span = p_tag.find('span', class_='c-review__prefix')
                    if prefix_span and 'c-review__prefix--color-green' in prefix_span.get('class', []):
                        contenu_positif = span_tags[0].text.strip().replace(" \n", ". ").replace("\n", ". ")
                    elif prefix_span:
                        contenu_negatif = span_tags[0].text.strip().replace(" \n", ". ").replace("\n", ". ")

            # Ajouter les données à la liste
            if not contenu_positif.startswith('«'):
                data['Code postal'].append(code_postal)
                data['Ville'].append(ville)
                data['Etoiles'].append(etoiles)
                data['Contenu positif'].append(contenu_positif)
                data['Contenu négatif'].append(contenu_negatif)
                data['Langue'].append(lang_attribut)
                data['Type'].append(bui_list_bodies)
                data['Nuits'].append(nuits)
                data['Date'].append(date)
                data['Pays'].append(pays)
                data['Occurrences'].append(1)
                data['Hôtel no'].append(no)

    return (data)



def click_and_scrape_hotel (url):
    executable_path = 'C:\\Users\\Chapatte\\OneDrive - VAUD PROMOTION\\Python pour Booking\\chromedriver-win64\\chromedriver.exe'
    service = Service(executable_path)
    options=webdriver.ChromeOptions()
    options.add_argument('--blink-settings=imagesEnabled=false')
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()
    driver.get(url)
    dataframe = {'Code postal': [],'Ville': [],'Etoiles': [], 'Contenu positif': [], 'Contenu négatif': [], 'Langue': [], 'Type': [], 'Nuits': [], 'Date': [], 'Pays': [], 'Occurrences': [], 'Hôtel no': []}
        
    if "js--hp-gallery-scorecard" in driver.page_source:
        print ("Au moins un commentaire pour ce logement")
        
        element_to_click = driver.find_element(By.ID,"js--hp-gallery-scorecard")
        element_to_click.click()
        wait = WebDriverWait(driver, 10)
        element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'review_list_new_item_block')))
        soup = bs(driver.page_source, 'html.parser')
        
        if "bui-pagination__link" in driver.page_source:
            page_blocks = soup.find_all('a', class_='bui-pagination__link')
            nbre_de_pages=[]
            for page_block in page_blocks:
                no_de_page = page_block.find_all('span')[0].get_text()
                nbre_de_pages.append(no_de_page)
            nombre_de_pages=int(nbre_de_pages[-1])

            for j in range(nombre_de_pages):
                time.sleep(1)
                wait = WebDriverWait(driver, 10)
                element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'review_list_new_item_block')))
                element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'bui-pagination__link')))
                element = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'bui-pagination__link')))
                scraped_page=scrape_page((j)*10,driver)
                next_page_to_click = driver.find_elements(By.CLASS_NAME,"bui-pagination__item")
                next_page_to_click[-1].click()
                dataframe=pd.concat([pd.DataFrame(dataframe),pd.DataFrame(scraped_page)])
        else:
            time.sleep(0.5)
            scraped_page=scrape_page(0,driver)
            dataframe=pd.DataFrame(scraped_page)

    else:
        print ("Aucun commentaire pour ce logement car js--hp-gallery-scorecard absent")
        dataframe=pd.DataFrame(dataframe)

    pd.options.display.max_columns=100
    return (dataframe)

etablissements_vaudois = pd.read_csv('Etablissements VD.csv', sep=";",encoding='utf-8')
#print the urls that are going to be scraped
print (etablissements_vaudois['url'][hotel_no:hotel_no+6])

comments_df = {'Code postal': [],'Ville': [],'Etoiles': [], 'Contenu positif': [], 'Contenu négatif': [], 'Langue': [], 'Type': [], 'Nuits': [], 'Date': [], 'Pays': [], 'Occurrences': [], 'Hôtel no': []}
comments_pd=pd.DataFrame(comments_df)
for no in range(hotel_no,hotel_no+6):
    print (no)
    hotel=etablissements_vaudois['url'][no]
    comments_of_hotel=click_and_scrape_hotel(hotel)
    comments_pd=pd.concat([comments_pd,comments_of_hotel])
print(comments_pd)


if comments_pd.empty:
    print("Tableau de données pandas vide, pas de csv enregistré.")
else:
    pre_existing_file = pd.read_csv('essai.csv', sep=";",encoding='utf-8')[1:]
    comments_pd=pd.concat([pre_existing_file,comments_pd])
    comments_pd.to_csv('essai.csv', encoding='utf-8',sep=';',index=False)
    print("csv enregistré.")