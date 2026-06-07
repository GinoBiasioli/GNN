
## Object hetero: SP2020 next activity with OC handover probability

Use this command on the VM to run the hyperparameter search first and then
immediately run the final repeated training runs for `sp2020` with
`oc_handover_probability`.

```powershell
git pull origin main
```

If the VM uses Linux/bash, activate the environment and install dependencies if
needed:

```bash
pip install -r requirements.txt
```

Hyperparameter search followed by final training:

```bash
python main_object_hetero.py sp2020 \
  --prediction-task next_activity \
  --graph-view oc_handover_probability \
  --graph-dir "gs://graphs-thesis/object hetero graphs" \
  --trials 10 \
  --epochs 50 \
  --runs 10
```

The script writes outputs under:

```text
results/sp2020/object_hetero/oc_handover_probability/
```

Expected bucket layout:

```text
gs://graphs-thesis/object hetero graphs/oc_handover_probability/sp2020/
  train_oc_handover.pt
  validation_oc_handover.pt
  test_oc_handover.pt
```
