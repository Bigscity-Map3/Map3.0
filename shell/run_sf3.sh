
for dataset in 'sanfransico'
do
    for model in 'START'
    do
      python run_model.py --task road_representation --model $model --dataset $dataset  --gpu_id 3 --config dim128_road_100 --exp_id $model
    done
done