#nohup sh run_road_all.sh JCLRNT 0 >run_poi_all.log 2>&1 &
for dataset in 'cd' 'bj'
do
    python run_model.py --task road_representation --model $1 --dataset $dataset --gpu_id $2 --config road_config_128 --exp_id $1 
done