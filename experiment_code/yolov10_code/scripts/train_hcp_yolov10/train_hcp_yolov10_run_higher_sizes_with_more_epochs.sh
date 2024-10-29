pct_arr=(10 25 50 75 100)
epochs=(200 150 100 100 100)

i=0
for pct in ${pct_arr[@]}
do
    echo "${pct} ${epochs[$i]}"
    sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/yolov10_code/scripts/train_hcp_yolov10/train_hcp_yolov10_run_subset.sbatch ${pct} ${epochs[$i]}
    i=$(( $i + 1));
done