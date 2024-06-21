
for dataset in 'chicago' 'foursquare_tky' 'singapore' 'porto' 'sanfransico'
do
    for model in 'CTLE' 
    do
      python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 0 --config config --exp_id $model
    done
done