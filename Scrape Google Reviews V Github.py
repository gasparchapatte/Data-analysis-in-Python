import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.wait import WebDriverWait
import pandas as pd
#from parsel import Selector
import urllib.parse

chromedrive_path = 'C:\\Users\\Chapatte\\OneDrive - VAUD PROMOTION\\Python pour Booking\\Chromedriver V 136\\chromedriver.exe'
driver = webdriver.Chrome(service=Service(chromedrive_path))

hotel_name = "YParc Hôtel"
query = urllib.parse.quote(hotel_name)
url = f"https://www.google.com/maps/search/{query}"
driver.get(url)

element_present = EC.presence_of_element_located((By.XPATH, '//span[text()="Tout accepter"]'))
try:
    WebDriverWait(driver, 15000).until(element_present).click()
except TimeoutException:
    print("Bouton pas trouvé.")

# Trouver le bouton avec le texte "Plus d'avis"
wait = WebDriverWait(driver, 10)
try:
    button = wait.until(EC.element_to_be_clickable(
    (By.XPATH, '//span[contains(@class, "wNNZR") and contains(@class, "fontTitleSmall") and contains(., "Plus d\'avis")]')))
    button.click()
    #print("✅ Bouton cliqué.")
except Exception as e:
    print("❌ Bouton non trouvé :", e)


#Faire défiler les commentaires et appuyer sur les boutons "plus"
while True:
    element = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, ".m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde")
    ))
    # Hauteur de l'élément scrollé avant le scroll
    scroll_top_before = driver.execute_script("return arguments[0].scrollTop;", element)

    # Scroller cet élément en bas (scrollHeight)
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", element)
    time.sleep(0.1)
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", element)
    time.sleep(0.1)
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", element)
    time.sleep(0.1)
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", element)
    time.sleep(0.1)

    #Cliquer sur les boutons "plus"
    while True:
        try:
            # Trouver tous les boutons présents
            buttons = wait.until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "button.w8nwRe.kyuRq")
            ))
            if not buttons:
                print("Plus aucun bouton à cliquer.")
                break
            
            # Cliquer sur le premier bouton de la liste
            button = buttons[0]
            button.click()

            # Attendre que la page/DOM ait le temps de se mettre à jour
            time.sleep(1)

        except Exception as e:
            print(f"Erreur rencontrée : {e}")
            break

    # Hauteur de l'élément scrollé après le scroll
    scroll_top_after = driver.execute_script("return arguments[0].scrollTop;", element)
    
    # Si le scroll ne bouge plus, on est arrivé en bas
    if scroll_top_after == scroll_top_before:
        print("🚩 Bas atteint, fin de la boucle")
        break

#Prend le contenu du commentaire
elements = driver.find_elements(By.CLASS_NAME, "MyEned")
for element in elements:
    try:
        # Chercher le span avec la classe wiI7pd à l'intérieur de cet élément
        span = element.find_element(By.CLASS_NAME, "wiI7pd")
        # Récupérer le texte à l'intérieur du span
        contenu = span.text
        print(contenu)
    except Exception as e:
        print(f"Span introuvable dans cet élément : {e}")

#Place le contenu du commmentaire dans un fichier Excel
contents = []

elements = driver.find_elements(By.CLASS_NAME, "MyEned")

for element in elements:
    try:
        span = element.find_element(By.CLASS_NAME, "wiI7pd")
        content = span.text
        contents.append(content)
    except Exception as e:
        print(f"Span introuvable dans cet élément : {e}")

df = pd.DataFrame(contents, columns=["Commentaires"])

# Chemin complet du fichier Excel (attention aux doubles backslashes ou raw string)
output_path = fr"C:\Users\Chapatte\OneDrive - VAUD PROMOTION\Python pour Google Reviews\Commentaires {hotel_name}.xlsx"

df.to_excel(output_path, index=False)

print(f"Données exportées dans {output_path}")