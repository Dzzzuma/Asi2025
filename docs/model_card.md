\# Model Card – AutoGluon Production Model (Sprint 3)



\## 1. Problem \& Intended Use



Celem modelu jest przewidywanie \*\*satysfakcji pasażerów linii lotniczych\*\* na podstawie informacji dotyczących:

\- doświadczenia podróży (obsługa, komfort, czas oczekiwania),

\- danych logistycznych (typ biletu, klasa podróży, długość podróży),

\- cech pasażera.



Model ma wspierać:

\- łatwiejszą analizę czynników wpływających na zadowolenie klientów,

\- szybkie porównywanie scenariuszy i segmentów klientów w badaniach jakości,

\- automatyczne wstępne klasyfikowanie opinii pasażerów w dashboardach analitycznych.



\*\*Model NIE powinien być używany jako samodzielna podstawa decyzji biznesowych,

a jedynie jako narzędzie wspomagające.\*\*



---



\## 2. Data (Source, License, Size, PII = No)



\*\*Źródło danych:\*\*  

Kaggle – \*Airline Passenger Satisfaction\*  

https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction  



\*\*Licencja:\*\*  

CC0 – Public Domain  



\*\*Data pobrania:\*\*  

6 października 2025 r.



\*\*Zakres danych:\*\*  

Oryginalny zbiór: ~130 000 rekordów  

Próbka w repozytorium: \*\*500 pierwszych wierszy\*\* (bez PII)



\*\*Dane wejściowe nie zawierają:\*\*

\- imion, nazwisk,

\- numerów rezerwacji,

\- dokumentów podróży,

\- lokalizacji.



Dane zawierają wyłącznie cechy ankietowe i logistyczne →  

\*\*oznaczamy je jako bezpieczne (PII = no).\*\*



---



\## 3. Metrics



Model oceniono na zbiorze testowym utworzonym w pipeline Kedro (deterministyczny split).



\### Główna metryka (kryterium wyboru Production)

\- \*\*ROC AUC:\*\* `0.91917`



Wybrano ROC AUC jako metrykę główną, ponieważ projekt dotyczy klasyfikacji binarnej

z potencjalną nierównowagą klas.



\### Metryki pomocnicze

\- \*\*Accuracy:\*\* `0.89`

\- \*\*F1 (weighted):\*\* `0.88997`



\### Produkcyjny run:

`ag\_exp2\_good\_quality` — osiągnął najlepszy wynik spośród trzech eksperymentów:



| Run                   | ROC AUC | Accuracy | F1 Weighted |

|----------------------|---------|----------|-------------|

| ag\_exp1\_fast         | niższe  | niższe   | niższe      |

| \*\*ag\_exp2\_good\_quality (production)\*\* | \*\*0.91917\*\* | \*\*0.89\*\* | \*\*0.88997\*\* |

| ag\_exp3\_high\_quality | niższe  | niższe   | niższe      |



Pełne porównanie:  

https://wandb.ai/s26282-pjatk/asi2025



---



\## 4. Limitations



\- Próbka (500 rekordów) jest mała w porównaniu do oryginalnych 130k rekordów — model może mieć wyższą wariancję.

\- Dane pochodzą z jednego okresu i jednej linii lotniczej → ograniczona generalizacja.

\- Satysfakcja pasażera jest subiektywna — metryki nie oddają w pełni złożoności problemu.

\- Możliwe ukryte biasy wynikające ze struktury danych:

&nbsp; - różnice między klasami podróży,

&nbsp; - różnice w typach biletów,

&nbsp; - różne preferencje grup wiekowych.

\- Modele AutoGluon (ensemble) mają ograniczoną interpretowalność.



---



\## 5. Ethics \& Risk



\### Ryzyka związane z danymi

\- Dane subiektywne — mogą odzwierciedlać indywidualne preferencje, nie obiektywną jakość usług.

\- Próbka może być niesymetryczna względem klas („Satisfied” vs „Dissatisfied”).

\- Mała próbka w repo może prowadzić do gorszej stabilności modelu.



\### Ryzyka związane z działaniem modelu

\- Możliwy drift danych przy zmianie polityki obsługi linii lotniczej.

\- Modele ensemble są trudne do wyjaśnienia → niska transparentność decyzji.

\- Model może przypadkowo wzmacniać uprzedzenia ukryte w danych.



\### Ryzyka etyczne

\- Możliwość dyskryminacji niektórych grup klientów (np. klas podróży, wieku).

\- Nadużywanie modelu do oceny pracowników lub jakości usług bez walidacji eksperckiej.



\### Sposoby minimalizacji ryzyka

\- Regularne retrenowanie modelu.

\- Monitoring driftu: zmiana dystrybucji cech → ponowne trenowanie.

\- Włączenie ekspertów („human in the loop”) przy interpretacji wyników.

\- Analiza feature importance / SHAP dla zwiększenia transparentności.



---



\## 6. Versioning



\- \*\*W\&B run (Production):\*\*  

&nbsp; https://wandb.ai/s26282-pjatk/asi2025/runs/t5qp34bo



\- \*\*Model artifact (production):\*\*  

&nbsp; https://wandb.ai/s26282-pjatk/asi2025/artifacts/model/ag\_model/v0  

&nbsp; Aliasy: `production`, `latest`, `candidate`



\- \*\*Code:\*\*  

&nbsp; Commit hash: `09ffce1`



\- \*\*Data:\*\*  

&nbsp; `airline\_passenger\_satisfaction\_sample.csv`  

&nbsp; (próbka 500 wierszy, Kaggle CC0)



\- \*\*Environment:\*\*  

&nbsp; Python 3.11  

&nbsp; AutoGluon 1.x  

&nbsp; scikit-learn 1.5  

&nbsp; Kedro 0.19.x  



\- \*\*Dashboard W\&B (porównanie modeli):\*\*  

&nbsp; https://wandb.ai/s26282-pjatk/asi2025



