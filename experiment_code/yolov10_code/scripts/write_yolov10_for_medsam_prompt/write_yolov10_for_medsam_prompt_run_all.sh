pct_arr=(0.1 0.5 1 2.5 5 10 25 50 75 100)
for pct in ${pct_arr[@]}
do
    sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/yolov10_code/scripts/write_yolov10_for_medsam_prompt/write_yolov10_for_medsam_prompt.sbatch ${pct}
done