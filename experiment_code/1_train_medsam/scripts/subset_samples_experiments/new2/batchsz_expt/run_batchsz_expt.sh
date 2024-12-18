label_arr=(71)
frac_arr=(0.1 100)
patience_arr=(3 3)
log_arr=(5 1)
batchsz_arr=(1 2 4 8 16 32 64 128 256)
#lr=(0.0005 0.001 0.005 0.01)
lr=(0.0005)
for u in ${lr[@]}
do
    for i in ${label_arr[@]}
    do
        idx=0
        for j in "${!frac_arr[@]}"
        do
            for batchsz in "${!batchsz_arr[@]}"
            do
                echo ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]} ${batchsz_arr[$batchsz]}
                workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/medsam_retrain_dice_batchsz_expt_11-8-24/training
                jobid1=$(sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/1_train_medsam/scripts/subset_samples_experiments/new2/batchsz_expt/train_subset_singletask_medsam_labels_dice_without_earlystopping_varying_batch.sbatch ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]} ${u} ${workdir} ${batchsz_arr[$batchsz]} | awk '{print $4}')

                workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/medsam_retrain_dice_batchsz_expt_11-8-24
                sbatch --time=02:00:00 --dependency=afterany:$jobid1 /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/batch_expt/run_subsample_medsam_5e-4_batch_expt.sbatch ${i} ${frac_arr[$j]} ${batchsz_arr[$batchsz]} ${workdir}
                let idx=${idx}+1
            done
        done
    done
done
