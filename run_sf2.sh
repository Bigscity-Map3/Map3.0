#nohup sh run2.sh >run2.log 2>&1 &
for dataset in 'sanfransico'
do
    for model in 'HyperRoad'
    do
      python run_model.py --task road_representation --model $model --dataset $dataset  --gpu_id 1 --config dim128_road_100 --exp_id $model
    done
done