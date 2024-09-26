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
            actual_date = data['rates'][0]['effectiveDate']  # Zwracamy rzeczywistą datę, dla której pobrano kurs
            rate = data['rates'][0]['mid']
            return rate, actual_date
        else:
            # Jeśli nie ma danych dla danego dnia, przesuwamy się o jeden dzień wstecz
            date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

# Funkcja obliczająca wartość portfela po 30 dniach
def calculate_portfolio_value(start_date, currencies, distribution, investment, days=30):
    # Daty rozpoczęcia i zakończenia inwestycji
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=days)
    
    # Pobieranie kursów walut na dzień rozpoczęcia i zakończenia wraz z rzeczywistymi datami
    rates_start = {currency: get_exchange_rate(currency, start_date.strftime("%Y-%m-%d")) for currency in currencies}
    rates_end = {currency: get_exchange_rate(currency, end_date.strftime("%Y-%m-%d")) for currency in currencies}
    
    # Obliczanie wartości początkowej w każdej walucie (ukryte z wyświetlania)
    initial_values = {currency: round((investment * dist) / rate_start[0], 2) for currency, dist, rate_start in zip(currencies, distribution, rates_start.values())}
    
    # Obliczanie wartości końcowej portfela w każdej walucie (ukryte z wyświetlania)
    final_values = {currency: round(initial_value * rate_end[0], 2) for currency, initial_value, rate_end in zip(initial_values.keys(), initial_values.values(), rates_end.values())}
    
    # Wyciąganie rzeczywistych dat
    start_dates_actual = {currency: rate_start[1] for currency, rate_start in rates_start.items()}
    end_dates_actual = {currency: rate_end[1] for currency, rate_end in rates_end.items()}
    
    # Obliczanie całkowitej wartości początkowej i końcowej portfela
    total_initial_value = round(investment, 2)
    total_final_value = round(sum(final_values.values()), 2)
    
    return rates_start, rates_end, total_initial_value, total_final_value, start_dates_actual, end_dates_actual, end_date

# Funkcja do generowania wykresów i ich zapisu do pliku
def generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, start_date, end_date):
    # Wykres podziału początkowego
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)  # zmienione z 1, 2, na 1, 3 (dodajemy jeden wykres)
    plt.pie(distribution, labels=currencies, autopct='%1.1f%%', startangle=140)
    plt.title('Początkowy podział inwestycji')
    
    # Wykres wartości portfela początkowego i końcowego
    plt.subplot(1, 3, 2)
    values = [total_initial_value, total_final_value]
    plt.bar(['Początek', 'Koniec'], values, color=['blue', 'green'])
    plt.title('Wartość portfela (PLN)')
    for index, value in enumerate(values):
        plt.text(index, value + 50, f'{value:.2f}', ha='center')  # Dodajemy wartość nad słupkami

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

# Użytkownik może wybrać datę startu (nie późniejszą niż dzisiaj - 30 dni)
start_date = st.date_input("Data startu", value=datetime.today() - timedelta(days=30), max_value=datetime.today() - timedelta(days=30))
usd_share = st.slider("USD %", min_value=0, max_value=100, value=30)
eur_share = st.slider("EUR %", min_value=0, max_value=100, value=40)
huf_share = st.slider("HUF %", min_value=0, max_value=100, value=30)

# Automatyczne obliczenie brakującego procenta
remaining_percentage = 100 - usd_share - eur_share - huf_share
if remaining_percentage != 0:
    st.warning(f"Procenty nie sumują się do 100%. Pozostało: {remaining_percentage}%.")

# Przycisk do uruchomienia analizy
if st.button("Uruchom analizę") and remaining_percentage == 0:
    investment = 1000  # stała kwota inwestycji
    currencies = ['usd', 'eur', 'huf']  # waluty
    distribution = [usd_share / 100, eur_share / 100, huf_share / 100]  # podział procentowy
    
    # Obliczanie wartości portfela
    rates_start, rates_end, total_initial_value, total_final_value, start_dates_actual, end_dates_actual, end_date = calculate_portfolio_value(start_date.strftime('%Y-%m-%d'), currencies, distribution, investment)
    
    # Prezentacja wyników i zapis wykresów
    generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, start_date, end_date)
    
    # Wyświetlanie danych
    st.write("Kursy na początku:", {currency: f"{rate[0]:.4f} ({start_dates_actual[currency]})" for currency, rate in rates_start.items()})
    st.write("Kursy na końcu:", {currency: f"{rate[0]:.4f} ({end_dates_actual[currency]})" for currency, rate in rates_end.items()})
    st.write(f"Wartość portfela na początku: {total_initial_value:.2f} PLN")
    st.write(f"Wartość portfela na końcu: {total_final_value:.2f} PLN")
