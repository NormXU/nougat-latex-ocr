# Nougat-LaTeX-OCR
  
Nougat-LaTeX-based is fine-tuned from [facebook/nougat-base](https://huggingface.co/facebook/nougat-base) with [im2latex-100k](https://zenodo.org/record/56198#.V2px0jXT6eA) to boost its proficiency in generating LaTeX code from images. 
Since the initial encoder input image size of nougat was unsuitable for equation image segments, leading to potential rescaling artifacts that degrades the generation quality of LaTeX code. To address this, Nougat-LaTeX-based adjusts the input resolution to a height of 224 and a width of 560. 
Additionally, an adaptive padding approach is used to ensure that equation image segments in the wild are resized to closely match the resolution of the training data.


### Evaluation
Evaluated on an image-equation pair dataset collected from Wikipedia, arXiv, and im2latex-100k, curated by [lukas-blecher](https://github.com/lukas-blecher/LaTeX-OCR#data)

|model| token_acc ↑ | normed edit distance ↓ |
| --- | --- | --- |
|pix2tex*|0.60|0.10|
|nougat-latex-based| **0.623850** | **0.06180** |

pix2tex*: reported from [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR); nougat-latex-based is evaluated on results generated with beam-search strategy. 

### Uses
1. Download model [here](https://huggingface.co/Norm/nougat-latex-base)
2. Install dependency
```bash
pip install -r all_requirements.txt
```
3. You can find an example in examples folder
```python
python examples/run_latex_ocr.py
```

### QA
- **Q:** Why did you copy and place the `image_processor_nougat.py` file in the repository rather than simply importing it from the `transformers` library if there are no changes compared to the one in `huggingface/transformers`?

- **A:** `transformers 4.34.0` is the first version that natively supports the nougat. However, there is a bug in the nougat processor within this version, which can result in a run failure. You can review the details of this issue [here](https://github.com/huggingface/transformers/issues/26597). Fortunately, the developers have already addressed this bug, and I anticipate that you will be able to directly import it from `transformers` in the next released version.
