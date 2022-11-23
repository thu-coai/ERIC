gpu=6
model=./model
target_name=./result/model_sample
output_suffix=stage2
#python3 -u ./gen2_gen.py $gpu $model test $target_name $output_suffix
python3 -u ./merge.py $target_name $output_suffix