# 🎯 EXECUTION GUIDE: Week 1 Complete Setup

## 🚀 EXECUTION: Follow These 5 Steps

### STEP 1: Read the Quick Start (2 minutes)
Open START_HERE.md

### STEP 2: Set Up Kaggle API (5 minutes)

You already generated the token! Now set it up:

```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move kaggle.json to the correct location
# You downloaded kaggle.json from Kaggle settings
# Put it in ~/.kaggle/

# On Mac/Linux - in terminal:
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# On Windows - in PowerShell:
mkdir $env:USERPROFILE\.kaggle
Copy-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
```

Verify it works:
```bash
kaggle datasets list | head
```

### STEP 3: Install Python Environment (5 minutes)

```bash
cd ml-financial-literacy-toolkit

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### STEP 4: Run the Pipeline (30-60 minutes)

```bash
# Option A: Automated (Recommended)
bash week1_pipeline.sh

# Option B: Manual (Step-by-step)
python src/utils/data_loader.py       # Download & load: 10-30 min
python src/utils/data_cleaner.py      # Clean data: 5-15 min
python notebooks/01_EDA.py            # Analysis: 5 min
```

### STEP 5: Review Results (5 minutes)

```bash
# Check cleaned data
ls -lh data/processed/

# View visualizations
ls results/visualizations/
```

## ⏱️ Timeline

| Step | Action | Time |
|------|--------|------|
| 1 | Read START_HERE.md | 2 min |
| 2 | Set up Kaggle API | 5 min |
| 3 | Install dependencies | 5 min |
| 4 | Run pipeline | 30-60 min |
| 5 | Review results | 5 min |
| **TOTAL** | **Complete Week 1** | **50-80 min** |

## ✅ Success Checklist

After completing all steps:

- [ ] Virtual environment created and activated
- [ ] All Python packages installed
- [ ] Kaggle API working (kaggle datasets list works)
- [ ] Data downloaded (~2GB)
- [ ] Data cleaned (2000+ stocks)
- [ ] 4 PNG visualizations generated
- [ ] Cleaning report (CSV) created
- [ ] You understand the data distribution

If all checks pass → **Week 1 COMPLETE!** 🎉

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `kaggle: command not found` | Activate venv: `source venv/bin/activate` |
| `Kaggle API error (401)` | Check ~/.kaggle/kaggle.json exists and is readable |
| `FileNotFoundError: cleaned_stocks...pkl` | Run data_loader.py then data_cleaner.py |
| `MemoryError` | Edit data_loader.py and change `limit=50` to `limit=20` |

## 🎓 What Comes Next (Week 2)

1. Feature Engineering - compute technical indicators
2. Pattern Labeling - create ground truth labels
3. Model Training - RF, XGBoost, K-Means
4. Evaluation - SHAP explanations
5. Phase 1 Report - document findings