# 🚀 START HERE: Week 1 Quick Start

Your entire project is ready! Here's how to execute it.

## ⚡ 5-Minute Quick Start

### 1. Install Python packages
```bash
cd ml-financial-literacy-toolkit
python -m venv venv
source venv/bin/activate  # Mac/Linux; Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up Kaggle API (one-time)
```bash
# You already have the token downloaded!
mkdir -p ~/.kaggle
# Copy the kaggle.json file you downloaded to ~/.kaggle/
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run the pipeline
```bash
# Option A: Automated (recommended)
bash week1_pipeline.sh

# Option B: Manual (step by step)
python src/utils/data_loader.py        # 10-30 min
python src/utils/data_cleaner.py       # 5-15 min  
python notebooks/01_EDA.py             # 5 min
```

Done! Check `results/visualizations/` for 4 PNG charts.

## 📖 Detailed Guides

- **[EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md)** ← Full walkthrough + troubleshooting
- **[WEEK1_CHECKLIST.md](./WEEK1_CHECKLIST.md)** ← Detailed checklist

## 🎯 What You'll Get

✅ 2000+ cleaned stocks  
✅ 4 visualization PNG files  
✅ Cleaning report (CSV)  
✅ Ready for Week 2 (feature engineering + modeling)

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Kaggle API error | Check ~/.kaggle/kaggle.json exists |
| FileNotFoundError | Run data_loader.py first |
| Out of memory | Reduce `limit` parameter in data_loader.py |

See WEEK1_CHECKLIST.md for more troubleshooting.

## Let's Go! 🚀

```bash
cd ml-financial-literacy-toolkit && bash week1_pipeline.sh
```