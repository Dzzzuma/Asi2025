# Model Card – AutoGluon Production Model (Sprint 3)

## 1. Problem & Intended Use

Celem modelu jest przewidywanie **satysfakcji pasażerów linii lotniczych** na podstawie informacji dotyczących:

- doświadczenia podróży (obsługa, komfort, czas oczekiwania),
- danych logistycznych (typ biletu, klasa podróży, długość podróży),
- cech pasażera.

Model ma wspierać:

- analizę czynników wpływających na zadowolenie klientów,
- porównywanie scenariuszy i segmentów klientów w badaniach jakości,
- automatyczne wstępne klasyfikowanie opinii pasażerów w dashboardach analitycznych.

**Model NIE powinien być używany jako samodzielna podstawa decyzji biznesowych, a jedynie jako narzędzie wspomagające.**

---

## 2. Data (Source, License, Size, PII = No)

**Źródło:**
Kaggle – Airline Passenger Satisfaction
https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction

**Licencja:** CC0
**Data pobrania:** 6 października 2025
**Rozmiar oryginału:** ~130 000 rekordów
**Próbka w repo:** 500 pierwszych wierszy
**PII:** brak danych osobowych (PII = no)

---

## 3. Metrics

Model oceniono na zbiorze testowym wygenerowanym w pipeline Kedro.

### Główna metryka:
- **ROC AUC:** `0.91917`

### Metryki pomocnicze:
- **Accuracy:** `0.89`
- **F1 (weighted):** `0.88997`

### Wybrany model produkcyjny
- Run: **ag_exp2_good_quality**
- Najwyższy wynik ROC AUC oraz stabilne accuracy/F1

Pełne porównanie:
https://wandb.ai/s26282-pjatk/asi2025

---

## 4. Limitations

- Próbka 500 rekordów może powodować większą wariancję modelu.
- Dane pochodzą z jednego źródła i mogą nie uogólniać się na inne linie lotnicze.
- Satysfakcja jest subiektywna, co utrudnia interpretację wyników.
- Możliwe ukryte uprzedzenia w danych (wiek, klasa podróży, typ biletu).
- Modele ensemble AutoGluon mają ograniczoną interpretowalność.

---

## 5. Ethics & Risk

### Ryzyka:
- Dane subiektywne mogą wprowadzać bias.
- Możliwość driftu danych w czasie.
- Trudna interpretowalność modeli ensemble.
- Ryzyko niesprawiedliwych decyzji biznesowych przy nadmiernym poleganiu na modelu.

### Mitigation:
- Regularne retrainingi.
- Monitoring driftu danych.
- Human-in-the-loop przy ważnych decyzjach.
- Analiza SHAP / feature importance.
- Kontrola jakości danych wejściowych.

---

## 6. Versioning

- **W&B run:**
  https://wandb.ai/s26282-pjatk/asi2025/runs/t5qp34bo

- **Model artifact:**
  https://wandb.ai/s26282-pjatk/asi2025/artifacts/model/ag_model/v0
  Aliasy: `production`, `latest`, `candidate`

- **Code (commit):**
  `09ffce1`

- **Data:**
  airline_passenger_satisfaction_sample.csv (Kaggle, 500 wierszy, CC0)

- **Environment:**
  Python 3.11
  AutoGluon 1.x
  scikit-learn 1.5
  Kedro 0.19.x

- **Dashboard:**
  https://wandb.ai/s26282-pjatk/asi2025
