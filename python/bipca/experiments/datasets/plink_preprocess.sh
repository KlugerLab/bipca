#!/bin/bash
plink2 --zst-decompress all_hg38.pgen.zst > all_hg38.pgen
plink2 --zst-decompress all_hg38.pvar.zst > all_hg38.pvar
plink2 --pfile all_hg38 --allow-extra-chr --max-alleles 2 --remove deg2_hg38.king.cutoff.out.id --chr 1-22 --maf 0.01 --make-bed -out all_phase3_out #--maf 0.1
plink --bfile all_phase3_out --indep 50 5 1.000005 #1.00005 #1.0005 #1.000005  # 1.005
plink --bfile all_phase3_out --extract plink.prune.in --make-bed -out all_phase3_pruned
