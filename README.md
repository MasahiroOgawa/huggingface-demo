# huggingface-demo
This is my first hugging face demo repository to test hugging face codes.


# How to run inference
- Set up the environment
```bash
conda env create -f environment.yml
conda activate huggingface-demo
```

- Run demo inference
```bash
python src/oneformer/run_oneformer.py
``` 

- RUn on your own image
```bash
python src/oneformer/run_oneformer.py [your image path]
```