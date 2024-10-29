pct_arr=(0.1 0.5 1 2.5 5 10 25 50 75 100)
tags=("train" "val" "test")
for pct in ${pct_arr[@]}
do
    for tag in ${tags[@]}
    do
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/yolov10_code/scripts/inference_hcp_yolov10/inference_hcp_yolov10_run_subset.sbatch ${pct} ${tag}
    done
done