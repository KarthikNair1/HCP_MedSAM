label_arr=(1 5 18 58)
frac_arr=(0.1 0.5 1 2.5 5 10 25 50 75 100)

for u in ${label_arr[@]}
do
    for i in ${frac_arr[@]}
    do
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/subsets_sample_experiments/run_subsample_medsam_1e-3_val.sbatch ${u} ${i}

    done
done
