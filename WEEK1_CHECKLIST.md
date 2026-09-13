# Week 1 Checklist: Data Loading, Cleaning & EDA

**Goal:** Acquire, clean, and analyze stock market data (35-60 minutes).

## 0️⃣ Pre-Setup: Kaggle API

You already have your API token! Now set it up:

1. You downloaded `kaggle.json` from Kaggle account settings
2. Put it in the correct location:
```bash
   mkdir -p ~/.kaggle
   cp ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
```
3. Verify it works:
```bash
   kaggle datasets list
```

## 1️⃣ Install Dependencies (5 min)

```bash
cd ml-financial-literacy-toolkit
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Verify:
```bash
python -c "import pandas, numpy, scikit-learn; print('✓ Success')"
```

## 2️⃣ Run Pipeline (30-50 min)

**Option A: Automated (Recommended)**
```bash
bash week1_pipeline.sh
```

**Option B: Step-by-Step**
```bash
python src/utils/data_loader.py       # 10-30 min
python src/utils/data_cleaner.py      # 5-15 min
python notebooks/01_EDA.py            # 5 min
```

## ✅ Success Checklist

- [ ] Virtual environment activated
- [ ] All packages installed (`pip list` shows 20+ packages)
- [ ] Kaggle API working (`kaggle datasets list` works)
- [ ] Data downloaded (~2GB in data/raw/)
- [ ] Data cleaned (2000+ stocks)
- [ ] 4 PNG visualizations in results/visualizations/
- [ ] Cleaning report CSV in data/processed/
- [ ] Can answer: What's the median trading days? Return distribution?

## 📊 Expected Outputs

- `data/processed/cleaned_stocks_sample.pkl` — Cleaned data
- `data/processed/cleaning_report.csv` — Quality metrics
- `results/visualizations/01_*.png` through `04_*.png` — 4 charts

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "kaggle: command not found" | Activate venv first |
| "Kaggle API error (401)" | Check ~/.kaggle/kaggle.json exists |
| "FileNotFoundError" | Run data_loader.py first |
| Memory error | Reduce limit in data_loader.py |
| Visualizations not showing | Check results/visualizations/ folder for PNG files |

## 🎯 What You'll Know After Week 1

- ✓ How many stocks in cleaned dataset?
- ✓ Typical date range and trading days?
- ✓ Return distribution shape?
- ✓ Are returns normally distributed?
- ✓ Volume patterns?