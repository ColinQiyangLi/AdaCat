# Vocoding with WaveNet and AdaCat

Data processing, training and evaluation proceed in stages indexed 0 through 3.
 - Stage 0: Data splitting (done once across models)
 - Stage 1: Feature generation and preprocessing (done once for continuous and once for discrete data)
 - Stage 2: Training
 - Stage 3: Evaluation

## Data processing
### Download LJSpeech data
Download data to `data/LJSpeech-1.1` from https://keithito.com/LJ-Speech-Dataset/. Make sure there is a `data/LJSpeech-1.1/wavs` subdirectory.
```
mkdir data
cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

### Process data
Generate data splits, stored in `data/split/lj`:
```
cd egs && ./run.sh mulaw256 0
```

Generate quantized features, stored in `egs/<subdir>/dump`:
```
# cat: Discrete, uniformly quantized features.
./run.sh cat30 1
./run.sh cat256 1
./run.sh cat512 1

# mulaw: Discrete, mulaw quantized features.
./run.sh mulaw30 1
./run.sh mulaw256 1
./run.sh mulaw512 1

# AdaCat: continuous features. Reuse for other experiments.
./run.sh adacat512 1
ln -s adacat512/dump adacat30/dump
ln -s adacat512/dump adacat256/dump

# Mixture of Logistics: continous features.
./run.sh mol30 1
ln -s mol30/dump mol255/dump
ln -s mol30/dump mol513/dump

# Gaussian: continuous features.
./run.sh gaussian 1
```

## Training
```
cd egs
./run.sh cat30 2
./run.sh cat256 2
./run.sh cat512 2
./run.sh mulaw30 2
./run.sh mulaw256 2
./run.sh mulaw512 2
./run.sh adacat30 2
./run.sh adacat256 2
./run.sh adacat512 2
./run.sh mol30 2
./run.sh mol255 2
./run.sh mol513 2
./run.sh gaussian 2
```

## Evaluate likelihood
Results will be be printed. The latest checkpoint in each experiment directory is used unless specified in the scripts' `eval_checkpoint` variable.
```
cd egs
./run_cat.sh cat30 3 3 --likelihood-only
./run_cat.sh cat256 3 3 --likelihood-only
./run_cat.sh cat512 3 3 --likelihood-only
./run_cat.sh mulaw30 3 3 --likelihood-only
./run_cat.sh mulaw256 3 3 --likelihood-only
./run_cat.sh mulaw512 3 3 --likelihood-only
./run_cat.sh adacat30 3 3 --likelihood-only
./run_cat.sh adacat256 3 3 --likelihood-only
./run_cat.sh adacat512 3 3 --likelihood-only
./run_cat.sh mol30 3 3 --likelihood-only
./run_cat.sh mol255 3 3 --likelihood-only
./run_cat.sh mol513 3 3 --likelihood-only
./run_cat.sh gaussian 3 3 --likelihood-only
```

## Generating samples
To generate audio samples, run the same commands as for likelihood evaluation, but omit the `--likelihood-only` flag.

## Acknowledgements
The codebase is based on Ryuichi Yamamoto's [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) repo.

