
for dataset in 'nyc' 'chicago' 'foursquare_tky' 'singapore' 'porto' 'sanfransico'
do
    for model in 'POI2Vec' 'Teaser' 'CTLE' 
    do
      python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 1 --config config --exp_id $model
    done
done