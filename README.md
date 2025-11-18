# asi-ml-template
Template repo do projektu

Źródło: Airline Passenger Satisfaction
Link: https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction
Licencja: CC0: Public Domain
Data pobrania: 6 października 2025r.
Info o próbce: do repo dodano 500 pierwszych wierszy z pelnego pliku (130k)

## Sprint 2 – pipeline data science
- przeniesiono logikę do `nodes.py`
- zdefiniowano pipeline: load → clean → split → train_baseline → evaluate

Model baseline i metryki są logowane do [Weights & Biases (W&B)](https://wandb.ai/sadej-jan-polsko-japo-ska-akademia-technik-komputerowych/asi2025).


## SPRINT 3 
Wybór modelu produkcyjnego (AutoGluon)

Do wyboru modelu „Production” przyjęliśmy następujące kryterium:
1. **Główna metryka**: maksymalna wartość ROC AUC na zbiorze testowym,
2. **Metryki pomocnicze**: accuracy oraz F1 (weighted) – model nie może mieć znacząco gorszych wartości niż pozostałe,
3. Wszystkie modele były trenowane na tej samej wersji danych (ten sam podział train/test z pipeline Kedro).

Na podstawie tych kryteriów wybraliśmy run:

- **Production run:** `ag_exp2_good_quality`  
  (wyższy ROC AUC oraz lepsze accuracy/F1 niż `ag_exp1_fast` i `ag_exp3_high_quality`).

  Szczegółowe wyniki (wykresy, pełny config runów) ->
https://wandb.ai/s26282-pjatk/asi2025

Wszystkie eksperymenty AutoGluon zostały wykonane na tej samej wersji danych:
- **Dane surowe:** `data/01_raw/airline_passenger_satisfaction_sample.csv`
- **Dane przetworzone:** artefakty generowane przez pipeline Kedro w `data/02_interim` i `data/03_processed`.
- **Podział train/test:** tworzony deterministycznie w nodzie `split_data` z ustawionym `random_state`.