label_arr=(1 5 18 58)
frac_arr=(0.1 0.5 1)

for u in ${label_arr[@]}
do
    for i in ${frac_arr[@]}
    do
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/subsets_sample_experiments/run_subsamples_unet_full_df_5e-4.sbatch ${u} ${i}
    done
done
