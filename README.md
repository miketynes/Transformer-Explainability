# JAX reproduction of results from [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838) [CVPR 2021]

Here we integrated our [jax reimplementation](https://github.com/miketynes/JAX-Transformer-Explainability) in the code that generated the results from Chefer *et al.*'s original work. We focused only on reproducing their results on BERT. 

### Our additions: 
* The relevant code from our jax reimplementation is found in the [BERT_flax](BERT_flax) directory. 
* The integration of this code with the BERT experiments is in [BERT_rationale_benchmark/models/pipeline/flax_bert_pipeline.py](BERT_rationale_benchmark/models/pipeline/flax_bert_pipeline.py)

The instructions for reproducing Chefer *et al.*'s BERT results are below, with additional instructions for producing our JAX results. 

Note that our reproduction was run on a single Polaris compute node (with access to a single GPU). 

## Reproducing results on BERT

1. Download the pretrained weights:

- Download `classifier.zip` from https://drive.google.com/file/d/1kGMTr69UWWe70i-o2_JfjmWDQjT66xwQ/view?usp=sharing
- mkdir -p `./bert_models/movies`
- unzip classifier.zip -d ./bert_models/movies/

2. Download the dataset pkl file:

- Download `preprocessed.pkl` from https://drive.google.com/file/d/1-gfbTj6D87KIm_u1QMHGLKSL3e93hxBH/view?usp=sharing
- mv preprocessed.pkl ./bert_models/movies

3. Download the dataset:

- Download `movies.zip` from https://drive.google.com/file/d/11faFLGkc0hkw3wrGTYJBr1nIvkRb189F/view?usp=sharing
- unzip movies.zip -d ./data/

4. Now you can run the model.

Example:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/models/pipeline/bert_pipeline.py --data_dir data/movies/ --output_dir bert_models/movies/ --model_params BERT_params/movies_bert.json
```
To control which algorithm to use for explanations change the `method` variable in `BERT_rationale_benchmark/models/pipeline/bert_pipeline.py` (Defaults to 'transformer_attribution' which is our method).
Running this command will create a directory for the method in `bert_models/movies/<method_name>`.

### Jax results
To reproduce the `jax` results, you run the above command but with the executable `flax_bert_pipeline.py`, like so: 
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/models/pipeline/flax_bert_pipeline.py --data_dir data/movies/ --output_dir bert_models/movies/ --model_params BERT_params/movies_bert.json
```

In order to run f1 test with k, run the following command:
```
PYTHONPATH=./:$PYTHONPATH python3 BERT_rationale_benchmark/metrics.py --data_dir data/movies/ --split test --results bert_models/movies/<method_name>/identifier_results_k.json
```

Also, in the method directory there will be created `.tex` files containing the explanations extracted for each example. This corresponds to our visualizations in the supplementary.

## Citing the original paper
The citation for the orginal work by Chefer *et al.*:
```
@InProceedings{Chefer_2021_CVPR,
    author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
    title     = {Transformer Interpretability Beyond Attention Visualization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {782-791}
}
```
