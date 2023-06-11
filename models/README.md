models version
- v1.0. = v1
  - hanya run biasa tanpa ada perubahan dari sumber
- v1.1
  - ubah scaller dari StandardScaler ke MinMaxScaler
  - tambah jumlah epoch dari 50 jadi 100


---

- v2.0
  - pendekatan baru dengan hanya menggunakan dense layer,dropout, dan normalization
  - minimal learning rate 1e-7
  - acuracy training 0.95 dan akurasi validasi 0.73
- v2.1
  - melanjutakn training untuk 50 epcoh
  - minimal learning rate 1e-10
  - acuracy training 0.96 dan akurasi validasi 0.738
- v2.2
  - melanjutakn training untuk 100 epcoh
  - minimal learning rate 1e-15 dan factor = 0.4
  - acuracy training 0.977 dan akurasi validasi 0.744