#Importuj biblioteki
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

#Pobieraj kurs waluty z API NBP dla danego dnia (brak kursów dla weekendów - pętla)
def get_exchange_rate(currency, date):
    while True:  
        url = f"http://api.nbp.pl/api/exchangerates/rates/A/{currency}/{date}/?format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            actual_date = data['rates'][0]['effectiveDate']  #Zwraca rzeczywistą datę
            rate = data['rates'][0]['mid']
            return rate, actual_date
        else:
            #Przesuwa się o jeden dzień wstecz
            date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

#Oblicza wartość portfela po 30 dniach
def calculate_portfolio_value(start_date, currencies, distribution, investment, days=30):
    #Początek i koniec inwestycji
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=days)
    
    #Pobiera kursy
    rates_start = {currency: get_exchange_rate(currency, start_date.strftime("%Y-%m-%d")) for currency in currencies}
    rates_end = {currency: get_exchange_rate(currency, end_date.strftime("%Y-%m-%d")) for currency in currencies}
    
    #Oblicza wartości początkowe w każdej walucie
    initial_values = {currency: round((investment * dist) / rate_start[0], 2) for currency, dist, rate_start in zip(currencies, distribution, rates_start.values())}
    
    #Oblicza wartości końcowe w każdej walucie
    final_values = {currency: round(initial_value * rate_end[0], 2) for currency, initial_value, rate_end in zip(initial_values.keys(), initial_values.values(), rates_end.values())}
    
    #Rzeczywista data
    start_dates_actual = {currency: rate_start[1] for currency, rate_start in rates_start.items()}
    end_dates_actual = {currency: rate_end[1] for currency, rate_end in rates_end.items()}
    
    #Oblicza całkowitą wartość na początku i końcu
    total_initial_value = round(investment, 2)
    total_final_value = round(sum(final_values.values()), 2)
    
    return rates_start, rates_end, total_initial_value, total_final_value, start_dates_actual, end_dates_actual, final_values, end_date

    #Wykresy
def generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, final_values, start_date, end_date):
    #Początek
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)  
    plt.pie(distribution, labels=currencies, autopct='%1.1f%%', startangle=140)
    plt.title('Początkowy podział inwestycji')
    
    #Porównanie
    plt.subplot(1, 3, 2)
    values = [total_initial_value, total_final_value]
    plt.bar(['Początek', 'Koniec'], values, color=['blue', 'green'])
    plt.title('Wartość portfela (PLN)')
    for index, value in enumerate(values):
        plt.text(index, value - 50, f'{value:.2f}', ha='center')

    #Koniec
    final_distribution = [final_values[currency] / total_final_value for currency in currencies]
    plt.subplot(1, 3, 3)
    plt.pie(final_distribution, labels=currencies, autopct='%1.1f%%', startangle=140)
    plt.title('Końcowy podział inwestycji')

    #Wykresy
    plt.suptitle(f'Inwestycja od {start_date} do {end_date}')  
    plt.tight_layout()
    plt.savefig("inwestycja_podsumowanie.png")  
    st.image("inwestycja_podsumowanie.png")  

#Streamlit
st.title("Analiza Portfela Inwestycyjnego")

#wybór daty startu (nie późniejszą niż dzisiaj minus 30 dni)
start_date = st.date_input("Data startu", value=datetime.today() - timedelta(days=30), max_value=datetime.today() - timedelta(days=30))
usd_share = st.slider("USD %", min_value=0, max_value=100, value=30)
eur_share = st.slider("EUR %", min_value=0, max_value=100, value=40)
huf_share = st.slider("HUF %", min_value=0, max_value=100, value=30)

#Podpowiedzi
remaining_percentage = 100 - usd_share - eur_share - huf_share
if remaining_percentage != 0:
    st.warning(f"Procenty nie sumują się do 100%. Pozostało: {remaining_percentage}%.")
    
#Przyciski
if st.button("Uruchom analizę") and remaining_percentage == 0:
    investment = 1000  #niezmienna
    currencies = ['usd', 'eur', 'huf']  
    distribution = [usd_share / 100, eur_share / 100, huf_share / 100]  
    
    #Obliczenia
    rates_start, rates_end, total_initial_value, total_final_value, start_dates_actual, end_dates_actual, final_values, end_date = calculate_portfolio_value(start_date.strftime('%Y-%m-%d'), currencies, distribution, investment)
    
    #Generuj
    generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, final_values, start_dates_actual[currencies[0]], end_date) 

    #Prezentuj
    st.write("Kursy na początku:", {currency: f"{rate[0]:.4f} ({start_dates_actual[currency]})" for currency, rate in rates_start.items()})
    st.write("Kursy na końcu:", {currency: f"{rate[0]:.4f} ({end_dates_actual[currency]})" for currency, rate in rates_end.items()})
    st.write(f"Wartość portfela na początku: {total_initial_value:.2f} PLN")
    st.write(f"Wartość portfela na końcu: {total_final_value:.2f} PLN")
