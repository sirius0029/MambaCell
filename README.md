# MambaCell: Multi-task Self-supervised Mamba for Cell Representation Learning
## Install
```bash
pip install -r requirements.txt
```

## Pretrained data
During pre-training, we utilize a large-scale pretraining corpus, Genecorpus-30M, comprising 27.4 million (27,406,217) human single-cell transcriptomes from a broad range of tissues from publicly available data. [download here](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)

## Downstream task data


The full descriptions of the datasets and the studies of origin can be found in the manuscript. Here we provide the links to access the processed datasets.

- PBMC 10K: [scvi_PBMC](https://docs.scvi-tools.org/en/stable/api/reference/scvi.data.pbmc_dataset.html)
- PBMC 3&86k: [PBMC 3&86k](https://www.10xgenomics.com/datasets)
- Immun Human dataset: [Immun Human dataset](https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/25717328/Immune_ALL_human.h5ad?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIYCQYOYV5JSSROOA/20250327/eu-west-1/s3/aws4_request&X-Amz-Date=20250327T014308Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=f28caa3e84eaf38ef32ec4237a2a607685bfbfc7acadd972f3364310906e022d)
- cardiomyopathy dataset: [cardiomyopathy](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset)
  
- Baron dataset: [Baron dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)

- Zheng 68k: [Zheng 68k](https://www.10xgenomics.com/datasets)

## Usage
For usage, see [examples](https://github.com/sirius0029/MambaCell/examples) for:
- pretraining
- cell type annotation
- batch integration
- disease classification

