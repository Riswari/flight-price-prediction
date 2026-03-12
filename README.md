# Flight Price Prediction

## Tech Stack
| Kategori | Tools |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Seaborn, Matplotlib |

---

## Model Performance
- **Algorithm:** Random Forest Regressor
- **Metric:** RMSLE
- **Test RMSLE:** 0.145237
- **MAE:** 1242 INR

---

## Dataset
Dataset berisi data harga tiket pesawat domestik India dengan fitur:
- `airline` — nama maskapai
- `source_city` — kota asal
- `destination_city` — kota tujuan
- `departure_time` — waktu keberangkatan
- `arrival_time` — waktu kedatangan
- `stops` — jumlah transit
- `class` — kelas penerbangan (Economy / Business)
- `duration` — durasi penerbangan (jam)
- `days_left` — hari sebelum keberangkatan
- `price` — harga tiket (INR) ← target

---

## Struktur Project
```
flight-price-prediction/
├── flight_price_prediction.ipynb
├── requirements.txt
└── README.md
```

---

## Author
**Rhey Indraswari**  
[GitHub](https://github.com/riswari)
