# Importujemy potrzebne biblioteki
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

# Funkcja pobierająca kurs waluty z API NBP dla danego dnia
def get_exchange_rate(currency, date):
    while True:  # Pętla sprawdzająca kurs, aż znajdziemy dane
        url = f"http://api.nbp.pl/api/exchangerates/rates/A/{currency}/{date}/?format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['rates'][0]['mid']
        else:
            # Jeśli nie ma danych dla danego dnia, przesuwamy się o jeden dzień wstecz
            print(f"Brak danych dla waluty {currency} na dzień {date}, próbuję dzień wcześniej.")
            date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

# Funkcja obliczająca wartość portfela po 30 dniach
def calculate_portfolio_value(start_date, currencies, distribution, investment, days=30):
    # Daty rozpoczęcia i zakończenia inwestycji
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=days)
    
    # Pobieranie kursów walut na dzień rozpoczęcia i zakończenia
    rates_start = {currency: get_exchange_rate(currency, start_date.strftime("%Y-%m-%d")) for currency in currencies}
    rates_end = {currency: get_exchange_rate(currency, end_date.strftime("%Y-%m-%d")) for currency in currencies}
    
    # Obliczanie wartości początkowej w każdej walucie
    initial_values = {currency: (investment * dist) / rates_start[currency] for currency, dist in zip(currencies, distribution)}
    
    # Obliczanie wartości końcowej portfela w każdej walucie
    final_values = {currency: initial_value * rates_end[currency] for currency, initial_value in initial_values.items()}
    
    # Obliczanie całkowitej wartości początkowej i końcowej portfela
    total_initial_value = investment
    total_final_value = sum(final_values.values())
    
    return rates_start, rates_end, initial_values, final_values, total_initial_value, total_final_value, end_date

# Funkcja do generowania wykresów i ich zapisu do pliku
def generate_plots(currencies, distribution, rates_start, rates_end, initial_values, final_values, total_initial_value, total_final_value, start_date, end_date):
    # Wykres podziału początkowego
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.pie(distribution, labels=currencies, autopct='%1.1f%%', startangle=140)
    plt.title('Początkowy podział inwestycji')
    
    # Wykres wartości portfela początkowego i końcowego
    plt.subplot(1, 2, 2)
    values = [total_initial_value, total_final_value]
    plt.bar(['Początek', 'Koniec'], values, color=['blue', 'green'])
    plt.title('Wartość portfela (PLN)')
    
    # Zapis wykresów do pliku PNG
    plt.suptitle(f'Inwestycja od {start_date} do {end_date.strftime("%Y-%m-%d")}')
    plt.tight_layout()
    plt.savefig("inwestycja_podsumowanie.png")  # Zapisujemy wykresy do pliku PNG
    st.image("inwestycja_podsumowanie.png")  # Wyświetlamy wykres w Streamlit

# Interfejs użytkownika w Streamlit
st.title("Analiza Portfela Inwestycyjnego")

start_date = st.date_input("Data startu", value=datetime.today())
usd_share = st.slider("USD %", min_value=0, max_value=100, value=30)
eur_share = st.slider("EUR %", min_value=0, max_value=100, value=40)
huf_share = st.slider("HUF %", min_value=0, max_value=100, value=30)

# Przycisk do uruchomienia analizy
if st.button("Uruchom analizę"):
    investment = 1000  # stała kwota inwestycji
    currencies = ['usd', 'eur', 'huf']  # waluty
    distribution = [usd_share / 100, eur_share / 100, huf_share / 100]  # podział procentowy
    
    # Obliczanie wartości portfela
    rates_start, rates_end, initial_values, final_values, total_initial_value, total_final_value, end_date = calculate_portfolio_value(start_date.strftime('%Y-%m-%d'), currencies, distribution, investment)
    
    # Prezentacja wyników i zapis wykresów
    generate_plots(currencies, distribution, rates_start, rates_end, initial_values, final_values, total_initial_value, total_final_value, start_date, end_date)
    
    # Wyświetlanie danych
    st.write("Kursy na początku:", rates_start)
    st.write("Kursy na końcu:", rates_end)
    st.write("Wartości początkowe:", initial_values)
    st.write("Wartości końcowe:", final_values)
    st.write(f"Wartość portfela na początku: {total_initial_value:.2f} PLN")
    st.write(f"Wartość portfela na końcu: {total_final_value:.2f} PLN")
