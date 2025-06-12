import requests
import pandas as pd
import snowflake.connector

# Paramètres pour l'API AirDNA
market_id = "35217"
url = "https://api.airdna.co/api/enterprise/v2/market/" + market_id + "/metrics/booking_lead_time"

payload = {
  "num_months": 24
}

headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Remplacer par le token
}

# Appel API
response = requests.post(url, json=payload, headers=headers)
data = response.json()

# Extraction des métriques
metrics = data['payload']['metrics']

# Liste pour stocker les lignes
rows = []

# Flattening des données
for metric in metrics:
    date = metric['date']
    for count in metric['reservation_counts']:
        lead_time_range = count['lead_time_day_range']
        num_reservations = count['num_reservations']
        
        # Assurez-vous que les valeurs NaN sont remplacées par None
        lead_time_start = int(lead_time_range[0])
        lead_time_end = int(lead_time_range[1]) if isinstance(lead_time_range[1], (int, float)) else 365
        
        rows.append({
            'date': date,
            'lead_time_start': lead_time_start,
            'lead_time_end': lead_time_end,
            'num_reservations': num_reservations
        })

# Convertir en DataFrame
df = pd.DataFrame(rows)
print(df)

# Connexion à Snowflake
conn = snowflake.connector.connect(
    user="gaspar",
    password="xxxxxxxxxxx",
    account="xxxxxxxxxxxxx-DEV_VAUDPROMOTION_DATAPLATEFORME",
    warehouse="COMPUTE_WH",
    database="TEST_DB",
    schema="TEST_SCHEMA"
)

# Préparer l'insertion des données
cursor = conn.cursor()

try:
    for index, row in df.iterrows():
        # Convertir la date au format approprié et insérer les entiers directement
        cursor.execute("""
            INSERT INTO booking_lead_time (reservation_date, lead_time_start, lead_time_end, num_reservations)
            VALUES (TO_DATE(%s, 'YYYY-MM-DD'), %s, %s, %s)
        """, (
            row['date'],
            row['lead_time_start'],
            row['lead_time_end'],
            row['num_reservations']
        ))
    
    print("Données insérées avec succès !")

except Exception as e:
    print(f"Erreur d'insertion : {e}")

finally:
    cursor.close()
    conn.close()