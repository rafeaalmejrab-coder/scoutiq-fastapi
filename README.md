# ScoutIQ FastAPI Backend

Backend API لتوقع نتائج مباريات كرة القدم باستخدام نموذج ScoutIQ الذي تم تدريبه في Kaggle.

## 1. تركيب المتطلبات

```bash
pip install -r requirements.txt
```

## 2. وضع ملفات النموذج

ضع مجلد:

```text
scoutiq_artifacts
```

بجانب ملف:

```text
app.py
```

ويجب أن يحتوي على:

```text
scoutiq_best_model.pkl
scoutiq_label_encoder.pkl
scoutiq_feature_columns.json
scoutiq_metadata.json
```

## 3. تشغيل السيرفر

```bash
uvicorn app:app --reload
```

ثم افتح:

```text
http://127.0.0.1:8000/docs
```

## 4. تجربة Endpoint

افتح `/docs` ثم جرّب `/predict`.

مثال JSON:

```json
{
  "home_team": "Man City",
  "away_team": "Brentford",
  "league": "E0",
  "country": "England",
  "neutral": 0,
  "home_elo": 1980,
  "away_elo": 1700,
  "elo_diff": 280,
  "home_last5_points_avg": 2.2,
  "away_last5_points_avg": 1.4,
  "home_last5_goals_for_avg": 2.1,
  "away_last5_goals_for_avg": 1.3,
  "home_last5_goals_against_avg": 0.9,
  "away_last5_goals_against_avg": 1.5,
  "home_last5_win_rate": 0.7,
  "away_last5_win_rate": 0.4,
  "last5_points_diff": 0.8,
  "last5_goals_for_diff": 0.8,
  "last5_goals_against_diff": -0.6,
  "last5_win_rate_diff": 0.3
}
```

## ملاحظة مهمة

هذا الـ API يتوقع بناءً على الخصائص التي ترسلها له.  
في النسخة القادمة يمكننا إضافة ملف `Matches.csv` داخل السيرفر لكي يحسب الخصائص تلقائياً بمجرد إرسال أسماء الفريقين فقط.
