## Install
```
pip install -r requirements.txt
```

## Build
Fill `spec.md` in with the contents of the most recent spec (the markdown "source" file):
```
python3 function_puller.py spec.md > spec.py
```

## Test
A short series of sanity check tests exist in `sanity_check.py`:
```
python3 sanity_check.py`
```

To output JSON test vectors of the sanity test:

```
python3 sanity_check.py --output-json
```

To output YAML test vectors of the sanity test:

```
python3 sanity_check.py --output-yaml
```
