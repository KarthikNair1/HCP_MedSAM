pct_arr=(0.1 0.5 1 2.5 5)
epochs=(440 440 360 330 303)

i=0
for pct in ${pct_arr[@]}
do
    echo "${pct} ${epochs[$i]}"
    sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/yolov10_code/scripts/train_hcp_yolov10/train_hcp_yolov10_run_subset.sbatch ${pct} ${epochs[$i]}
    i=$(( $i + 1));
done