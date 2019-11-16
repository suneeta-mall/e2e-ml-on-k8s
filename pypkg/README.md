# pypkg

Python package containing core functions for e2e ml [example](https://github.com/suneeta-mall/e2e-ml-on-k8s).

```bash
# pip install pypkg
cd pypkg; 
noglob pip uninstall pylib --yes;
python3 setup.py sdist bdist_wheel
pip install --no-cache-dir dist/pylib-1.0.0-py3-none-any.whl
rm -rf build dist pylib.egg-info
cd ..
```