#!/bin/bash
/data/jyc/software/plink2 --zst-decompress all_hg38.pgen.zst > all_hg38.pgen
/data/jyc/software/plink2 --zst-decompress all_hg38.pvar.zst > all_hg38.pvar
/data/jyc/software/plink2 --pfile all_hg38 --allow-extra-chr --max-alleles 2 --remove deg2_hg38.king.cutoff.out.id --chr 1-22 --maf 0.1 --make-bed -out all_phase3_out #--maf 0.01
/data/jyc/software/plink --bfile all_phase3_out --indep 200 5 1.005  #1.000005 #1.00005 #1.0005 #1.000005
/data/jyc/software/plink --bfile all_phase3_out --extract plink.prune.in --make-bed -out all_phase3_pruned
