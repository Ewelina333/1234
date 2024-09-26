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
            return data['rates'][0]['mid'], date
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
    initial_values = {currency: (investment * dist) / rates_start[currency][0] for currency, dist in zip(currencies, distribution)}
    
    # Obliczanie wartości końcowej portfela w każdej walucie
    final_values = {currency: initial_value * rates_end[currency][0] for currency, initial_value in initial_values.items()}
    
    # Obliczanie całkowitej wartości początkowej i końcowej portfela
    total_initial_value = investment
    total_final_value = sum(final_values.values())
    
    return rates_start, rates_end, total_initial_value, total_final_value, end_date

# Funkcja do generowania wykresów i ich zapisu do pliku
def generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, final_values, start_date, end_date):
    # Wykres podziału początkowego
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)  # zmienione z 1, 2, na 1, 3 (dodajemy jeden wykres)
    plt.pie(distribution, labels=currencies, autopct='%1.1f%%', startangle=140)
    plt.title('Początkowy podział inwestycji')
    
    # Wykres wartości portfela początkowego i końcowego
    plt.subplot(1, 3, 2)
    values = [total_initial_value, total_final_value]
    bars = plt.bar(['Początek', 'Koniec'], values, color=['blue', 'green'])
    plt.title('Wartość portfela (PLN)')

    # Dodajemy wartości nad słupkami, ale nieco niżej, aby były lepiej widoczne
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval - 100, f'{yval:.2f}', ha='center', va='bottom', color='white')

    # Wykres procentowego podziału po 30 dniach (dodany nowy wykres)
    final_distribution = [final_values[currency] / total_final_value for currency in currencies]
    plt.subplot(1, 3, 3)
    plt.pie(final_distribution, labels=currencies, autopct='%1.1f%%', startangle=140)
    plt.title('Końcowy podział inwestycji')

    # Zapis wykresów do pliku PNG
    plt.suptitle(f'Inwestycja od {start_date} do {end_date.strftime("%Y-%m-%d")}')
    plt.tight_layout()
    plt.savefig("inwestycja_podsumowanie.png")  # Zapisujemy wykresy do pliku PNG
    st.image("inwestycja_podsumowanie.png")  # Wyświetlamy wykres w Streamlit

# Interfejs użytkownika w Streamlit
st.title("Analiza Portfela Inwestycyjnego")

# Data input with a limit for max date (today - 30 days)
today = datetime.today()
start_date = st.date_input("Data startu", value=today - timedelta(days=30), max_value=today - timedelta(days=30))

usd_share = st.slider("USD %", min_value=0, max_value=100, value=30)
eur_share = st.slider("EUR %", min_value=0, max_value=100, value=40)
huf_share = st.slider("HUF %", min_value=0, max_value=100, value=30)

# Wyświetlanie informacji o pozostałych procentach
remaining_percentage = 100 - (usd_share + eur_share + huf_share)
st.write(f"Pozostałe do rozdzielenia: {remaining_percentage}%")

# Zabezpieczenie: suma udziałów musi wynosić 100%
if usd_share + eur_share + huf_share != 100:
    st.error("Łączna suma udziałów walut musi wynosić 100%.")
else:
    # Przycisk do uruchomienia analizy
    if st.button("Uruchom analizę"):
        investment = 1000  # stała kwota inwestycji
        currencies = ['usd', 'eur', 'huf']  # waluty
        distribution = [usd_share / 100, eur_share / 100, huf_share / 100]  # podział procentowy
        
        # Obliczanie wartości portfela
        rates_start, rates_end, total_initial_value, total_final_value, end_date = calculate_portfolio_value(start_date.strftime('%Y-%m-%d'), currencies, distribution, investment)
        
        # Prezentacja wyników i zapis wykresów
        final_values = {currency: (investment * dist) / rates_start[currency][0] * rates_end[currency][0] for currency, dist in zip(currencies, distribution)}
        generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, final_values, start_date, end_date)
        
        # Wyświetlanie danych z krótkim wyjaśnieniem
        st.write(f"Kursy na początku: {rates_start}")
        st.write(f"Kursy na końcu: {rates_end}")
        st.write(f"Wartość portfela na początku: {total_initial_value:.2f} PLN (suma początkowej wartości w PLN)")
        st.write(f"Wartość portfela na końcu: {total_final_value:.2f} PLN (suma końcowej wartości w PLN)")


