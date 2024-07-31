
#nohup sh run_poi_all.sh >run_poi_all.log 2>&1 &
for dataset in 'nyc' 'foursquare_tky'  'chicago' 'singapore' 'porto' 'sanfransico'
do
    for model in 'CTLE' 'Hier' 'POI2Vec' 'SkipGram' 'Teaser' 'Tale' 'CACSR'
    # for model in 'POI2Vec'
    do
      python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 1 --config poi_config_128 --exp_id $model --train false
    done
done

# python run_model.py --task region_representation --model $1 --dataset $2 --config dim128_region_100 --train false --exp_id $1