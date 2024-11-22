label_arr=(1 5 7)
frac_arr=(0.1 100)
batch_arr=(1 2 4 8 16 32 64 128 256)

for u in ${label_arr[@]}
do
    for i in ${frac_arr[@]}
    do
        for bs in ${batch_arr[@]}
        do
            export outdir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/medsam_retrain_dice_batchsz_expt_11-8-24'
            sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/batch_expt/run_subsample_medsam_5e-4_batch_expt.sbatch ${u} ${i} ${bs} ${outdir}

            #export outdir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_retrain_dice_batchsz_expt_11-11-24'
            #sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/batch_expt/run_subsample_medsam_5e-4_batch_expt.sbatch ${u} ${i} ${bs} ${outdir}
        done
    done
done
