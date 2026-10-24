# Day 2-3 Lab: Answers

Name:

Numbers come from `python -m quakes.etl` (2084 events in `quakes.db`, 7.9% big quakes) and the `quakes.train` reports in `reports/`.

## 1. Baseline
How much does your regression beat "predict the mean" (RMSE and R²)? Is it worth deploying?

On the test set, `--model linear` gets **RMSE 0.476 vs 1.299** for the mean baseline, and **R² 0.865 vs −0.007**. The cross-validated R² on train is 0.869 ± 0.011, so this isn't a lucky split. That's a clear win, but most of it comes from *who* recorded the event, not from the physics. Dropping `net` lowers R² to 0.755, and dropping the station-geometry columns (`dmin`, `gap`, `nst`, `rms`) lowers it to 0.803: the global `us` network reports mainly large, distant events, while local networks report small nearby ones. So the model works as a quick estimate inside this feed, but I wouldn't deploy it as a magnitude predictor without checking it on a different time window and network mix. An error of about ±0.5 magnitude units is also large near the 4.5 cut-off.

## 2. Accuracy paradox
What accuracy and recall does the dummy classifier get on `big_quake`? Why is accuracy the wrong headline metric?

The majority-class dummy gets **92.1% accuracy with 0.0 recall** (precision 0, F1 0, ROC-AUC 0.5). It never predicts a big quake, yet it looks "92% right" because only ~8% of events are positive. Accuracy rewards the majority class, so for a rare, costly event the headline metrics should be recall, precision/F1 and ROC-AUC. On those, logistic regression (recall 0.82, F1 0.82, AUC 0.99) is far better than the dummy, while its accuracy only rises from 0.92 to 0.97.

## 3. Threshold
What threshold did your search choose, compared with 0.5? Which error would you rather make in an earthquake alert system, and why?

Searching for the best F1 on **out-of-fold training probabilities** (`cross_val_predict`; the test set is used only once, at the end) chose **0.40**. On test, recall went from 0.818 to 0.848, precision went from 0.818 to 0.800, and F1 went from 0.818 to 0.824. In an alert system I would rather raise a **false alarm** than miss a real M4.5+ event. A missed alert can cost lives, while a false alarm costs trust and some disruption. So I'd push the threshold lower still (or search on recall) and accept more false positives, as long as the false-alarm rate stays tolerable. With `--balanced` the model already reaches recall 0.94 at 0.5, and the search then picks 0.85 to win precision back.

## 4. Leakage check
Remove one leaky column name from `config.LEAKY`, rerun the ETL and train, and compare scores. What happened, and why is the better number a lie?

I removed `sig` from `LEAKY` (and added it to `NUMERIC` so the model can use it) and reran the same pipeline:

| | R² | RMSE | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| honest (`sig` dropped) | 0.871 | 0.472 | 0.818 | 0.794 | 0.9941 |
| `sig` kept | **0.946** | **0.306** | **0.939** | **0.954** | **0.9994** |

Every score jumped, but it's a lie. USGS computes `sig` *from* the magnitude (plus felt reports and PAGER alerts), so the model is reading the answer back from a column built from the target. When a new event comes in, `sig` doesn't exist until after the magnitude is known, so this model could never be used for prediction. Restoring `sig` to `LEAKY` brings the honest numbers back exactly.

## 5. Complexity
Train with `--model poly` at degree 1, 2 and 4. When does performance on the test set fall below the training result?

| degree | train R² | CV R² (train) | test R² | test RMSE |
|---|---|---|---|---|
| 1 | 0.879 | 0.869 | 0.865 | 0.475 |
| 2 | 0.917 | 0.855 | **0.888** | **0.434** |
| 4 | 0.972 | −7669 | **−18673** | 176.8 |

Test R² is a little below train at every degree, which is normal. The gap stays small at degree 1, grows at degree 2 (train 0.917 vs CV 0.855, though test is still the best of the three), and blows up at degree 4. With 14 numeric features, degree 4 creates thousands of interaction terms. Train R² keeps rising (0.97), but on unseen rows the extreme polynomial values extrapolate wildly and the predictions are useless. The cross-validation on train already shows it (−7669), so I can reject degree 4 without ever looking at the test set. Degree 2 is the sweet spot.

## 6. Gradient descent
Do your `GDLinearRegression` coefficients match `LinearRegression` on the notebook data? What changes if you skip scaling?

Yes. On the synthetic notebook data (`lr=0.1`, 1000 iterations), GD gives `[1.956, −0.989, 0.520]`, identical to `LinearRegression` to three decimals. The loss falls steadily, and the vectorized gradient is about 200× faster than the loop version. Without scaling, the features differ in size by 500× (×50 vs ×0.1). The learning rate must be tiny (1e‑4) to stop the large feature from diverging, and then the small feature barely moves. After 500 iterations the loss was **6.53 unscaled vs 0.13 scaled**, so GD hadn't converged. Scaling makes the loss surface round, so one learning rate suits every weight.
