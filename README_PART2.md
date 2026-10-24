# Day 2-3 Lab: Part 2 additions

Copy these files into your Day 1 project (same folder structure). Then update your environment once:

```bash
conda env update -f environment.yml --prune
```

| File | Task |
|---|---|
| `quakes/etl.py` | A: ETL with logging |
| `quakes/models.py` | B: estimator and model factories (also edit `transform.build_preprocessor`) |
| `quakes/evaluate.py` | C: metrics |
| `quakes/train.py` | D: training command line |
| `notebooks/day2-3_part1_sklearn_basics.ipynb` | Part 1 (do this first) |

```bash
pytest tests/test_etl.py        # Task A
pytest tests/test_models.py     # Task B
pytest tests/test_evaluate.py   # Task C
python -m quakes.etl --mode batch
python -m quakes.train --task regression --model linear
python -m quakes.train --task classification --model logistic
```

Create `ANSWERS_ML.md` with your answers to the six lab questions in the handout.

## Scheduling note (write in this file)
Add the exact cron line and the Windows Task Scheduler steps that would run
`python -m quakes.etl --mode batch` every hour.

### Linux / macOS (cron)
`crontab -e`, then add this line (it runs at minute 0 of every hour, from the project root, with the env's Python):
```cron
0 * * * * cd /home/<user>/day1-quake-preprocessing && /home/<user>/anaconda3/envs/niai-ml/bin/python -m quakes.etl --mode batch >> logs/cron.out 2>&1
```

### Windows (Task Scheduler)
1. Open **Task Scheduler** → **Create Task…** (not "Basic Task"), name it `quakes-etl`.
2. **Triggers** → New → *Daily*, start now, tick **Repeat task every: 1 hour**, *for a duration of: Indefinitely*.
3. **Actions** → New → *Start a program*:
   - Program/script: `C:\Users\WaleedsPC\anaconda3\envs\niai-ml\python.exe`
   - Add arguments: `-m quakes.etl --mode batch`
   - Start in: `C:\Users\WaleedsPC\Desktop\day1-quake-preprocessing` (required, or `import quakes` fails)
4. **Settings** → tick *Run task as soon as possible after a scheduled start is missed*, and *If the task fails, restart every 5 minutes, up to 3 times*.

The same task from a terminal:
```
schtasks /Create /TN quakes-etl /SC HOURLY /MO 1 /TR "cmd /c cd /d C:\Users\WaleedsPC\Desktop\day1-quake-preprocessing && C:\Users\WaleedsPC\anaconda3\envs\niai-ml\python.exe -m quakes.etl --mode batch"
```

### Where the log goes and how to spot a failed run
- Every run appends to **`logs/etl.log`** (project root): row counts after extract, transform and load, and the table size before and after.
- A failed run writes an `ERROR ... ETL failed` line with the traceback and exits with **code 1**. Task Scheduler shows that as *Last Run Result: 0x1* (success is `0x0`), and cron's `logs/cron.out` captures the same traceback.
- A `WARNING ... using data/fallback/all_week.geojson` line means the network failed and the run used stale data. That isn't a crash, but it's worth checking.
- If the newest `ETL done` line in `etl.log` is older than an hour or two, the schedule itself has stopped running.
