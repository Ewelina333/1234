# Importujemy potrzebne biblioteki
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display

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
    plt.figure(figsize=(10,5))
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
    plt.show()

# Funkcja uruchamiana przez użytkownika w celu wprowadzenia danych
def run_analysis(start_date, usd_share, eur_share, huf_share):
    investment = 1000  # stała kwota inwestycji
    currencies = ['usd', 'eur', 'huf']  # waluty
    distribution = [usd_share / 100, eur_share / 100, huf_share / 100]  # podział procentowy
    
    # Obliczanie wartości portfela
    rates_start, rates_end, initial_values, final_values, total_initial_value, total_final_value, end_date = calculate_portfolio_value(start_date, currencies, distribution, investment)
    
    # Prezentacja wyników i zapis wykresów
    generate_plots(currencies, distribution, rates_start, rates_end, initial_values, final_values, total_initial_value, total_final_value, start_date, end_date)
    
    # Wyświetlanie danych
    print("Kursy na początku:", rates_start)
    print("Kursy na końcu:", rates_end)
    print("Wartości początkowe:", initial_values)
    print("Wartości końcowe:", final_values)
    print(f"Wartość portfela na początku: {total_initial_value:.2f} PLN")
    print(f"Wartość portfela na końcu: {total_final_value:.2f} PLN")

# Interfejs użytkownika za pomocą suwaków
start_date_picker = widgets.DatePicker(description='Data startu', value=datetime.today())
usd_slider = widgets.IntSlider(description='USD %', value=30, min=0, max=100)
eur_slider = widgets.IntSlider(description='EUR %', value=40, min=0, max=100)
huf_slider = widgets.IntSlider(description='HUF %', value=30, min=0, max=100)

# Aktualizowanie suwaków, aby sumowały się do 100%
def update_sliders(change):
    total = usd_slider.value + eur_slider.value + huf_slider.value
    if total != 100:
        if change['owner'] == usd_slider:
            eur_slider.value = max(0, min(100, 100 - usd_slider.value - huf_slider.value))
        elif change['owner'] == eur_slider:
            usd_slider.value = max(0, min(100, 100 - eur_slider.value - huf_slider.value))
        elif change['owner'] == huf_slider:
            eur_slider.value = max(0, min(100, 100 - usd_slider.value - huf_slider.value))

usd_slider.observe(update_sliders, names='value')
eur_slider.observe(update_sliders, names='value')
huf_slider.observe(update_sliders, names='value')

# Przycisk do uruchomienia analizy
button = widgets.Button(description="Uruchom analizę")
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        run_analysis(start_date_picker.value.strftime('%Y-%m-%d'), usd_slider.value, eur_slider.value, huf_slider.value)

button.on_click(on_button_click)

# Wyświetlenie suwaków i przycisku
display(start_date_picker, usd_slider, eur_slider, huf_slider, button, output)
