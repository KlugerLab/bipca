#!/bin/bash

allcools generate-dataset \
--allc_table allc_table.tsv \
--output_path RufZamojski2021NC.mcds \
--chrom_size_path mm10.main.chrom.sizes \
--obs_dim cell \
--cpu 10 \
--chunk_size 50 \
--regions chrom5k 5000 \
--quantifiers chrom5k count CGN \
