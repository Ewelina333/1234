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
            return data['rates'][0]['mid'], date  # Zwracamy kurs i datę
        else:
            # Jeśli nie ma danych dla danego dnia, przesuwamy się o jeden dzień wstecz
            date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

# Funkcja obliczająca wartość portfela po 30 dniach
def calculate_portfolio_value(start_date, currencies, distribution, investment, days=30):
    # Daty rozpoczęcia i zakończenia inwestycji
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=days)
    
    # Pobieranie kursów walut na dzień rozpoczęcia i zakończenia
    rates_start = {currency: get_exchange_rate(currency, start_date.strftime("%Y-%m-%d")) for currency in currencies}
    rates_end = {currency: get_exchange_rate(currency, end_date.strftime("%Y-%m-%d")) for currency in currencies}
    
    # Obliczanie całkowitej wartości początkowej i końcowej portfela
    total_initial_value = investment
    final_values = {currency: (investment * dist / rates_start[currency][0]) * rates_end[currency][0] for currency, dist in zip(currencies, distribution)}
    total_final_value = sum(final_values.values())
    
    return rates_start, rates_end, total_initial_value, total_final_value, end_date

# Funkcja do generowania wykresów i ich zapisu do pliku
def generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, start_date, end_date):
    # Wykres podziału początkowego
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.pie(distribution, labels=currencies, autopct='%1.1f%%', startangle=140)
    plt.title('Początkowy podział inwestycji')
    
    # Wykres wartości portfela początkowego i końcowego
    plt.subplot(1, 3, 2)
    values = [total_initial_value, total_final_value]
    plt.bar(['Początek', 'Koniec'], values, color=['blue', 'green'])
    plt.title('Wartość portfela (PLN)')
    plt.text(0, values[0], f'{values[0]:.2f}', ha='center', va='bottom', fontsize=12, color='black')
    plt.text(1, values[1], f'{values[1]:.2f}', ha='center', va='bottom', fontsize=12, color='black')
    
    # Obliczanie końcowego procentowego podziału inwestycji
    final_distribution = [rates_end[currency][0] / total_final_value for currency in currencies]
    
    # Wykres końcowego procentowego podziału
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

# Ustawienie zakresu dat: maksymalna data to dzisiaj - 30 dni
max_date = datetime.today() - timedelta(days=30)
start_date = st.date_input("Data startu (nie później niż 30 dni temu)", value=max_date, max_value=max_date)

usd_share = st.slider("USD %", min_value=0, max_value=100, value=30)
eur_share = st.slider("EUR %", min_value=0, max_value=100, value=40)
huf_share = st.slider("HUF %", min_value=0, max_value=100, value=30)

# Podpowiedź ile procent jeszcze brakuje
total_share = usd_share + eur_share + huf_share
remaining_share = 100 - total_share
st.write(f"Pozostały procent do rozdysponowania: {remaining_share}%")

# Przycisk do uruchomienia analizy
if total_share == 100:
    if st.button("Uruchom analizę"):
        investment = 1000  # stała kwota inwestycji
        currencies = ['usd', 'eur', 'huf']  # waluty
        distribution = [usd_share / 100, eur_share / 100, huf_share / 100]  # podział procentowy

        # Obliczanie wartości portfela
        rates_start, rates_end, total_initial_value, total_final_value, end_date = calculate_portfolio_value(
            start_date.strftime('%Y-%m-%d'), currencies, distribution, investment)

        # Prezentacja wyników i zapis wykresów
        generate_plots(currencies, distribution, rates_start, rates_end, total_initial_value, total_final_value, start_date, end_date)

        # Wyświetlanie danych z krótkim wyjaśnieniem w oddzielnych akapitach
        st.write("Kursy walut na początku i na końcu okresu inwestycji:")

        for currency in currencies:
            start_rate, start_date = rates_start[currency]
            end_rate, end_date = rates_end[currency]
            
            st.write(f"**Waluta: {currency.upper()}**")
            st.write(f"- Kurs początkowy ({start_date}): {start_rate:.4f} PLN")
            st.write(f"- Kurs końcowy ({end_date}): {end_rate:.4f} PLN")
            st.write("---")

        # Wyświetlanie wartości portfela na początku i końcu
        st.write(f"Wartość portfela na początku: {total_initial_value:.2f} PLN (suma początkowej wartości w PLN)")
        st.write(f"Wartość portfela na końcu: {total_final_value:.2f} PLN (suma końcowej wartości w PLN)")

else:
    st.warning("Suma udziałów musi wynosić 100%. Proszę dostosować suwak.")



