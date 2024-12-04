#!/bin/bash
# seq_dim=(2 4 8 10)
seq_dim_all=(4)
name_all=("svm" "dt" "gbdt")
# name_all=("svm")


python main.py    --data_dir=./data/  \
                  --scale=scale \
                  --fields_file=./ids_fields.txt \
                  --fingerprint \
                  --train \
                  --model_path=./models_save \
                  --model_name=fingerprint_ids \
                  --save_path=./results \
                  --save_name=fingerprint_ids \
                  --input_dim=16,16,16,16,16,16,16,16 \
                  --encoding_dim=8,8,8,8,8,8,8,8 \
                  --seq_dim=32,32,32,32,32,32,32,32 \
                  --n_epochs=100,50,100,100,50,100,50,50 \
                  --debug=True


#
#for name in "${name_all[@]}"; do
#    for seq_dim in "${seq_dim_all[@]}"; do
#            new_name=${name}"_seqdim_"${seq_dim}"_kl_final"
#            current_time=$(date "+%Y-%m-%d %H:%M:%S")
#             echo "********当前时间：$current_time 当前算法名称：$name---当前seqdim：$seq_dim-----当前模式：train********"
#             python main.py  --$name \
#                             --train \
#                             --selected \
#                             --data_dir /data/dataset/process_data_geeksec/selected15/npy \
#                             --class_choice IRCbot,Menti,Murlo,Neris,Osx-trojan,RBot,Smokebot,Sogou,Weasel,benign,convagent,covenant,ddosfamily-tcp-thc-ssl,dos-hulk,dos-slowhttptest,emotet,malware-dridex,malware-ursnif,poshc2,qakbot,thefatrat,trickbot \
#                             --scale scale \
#                             --seq_dim $seq_dim \
#                             --model_path ./models/geeksec/$new_name \
#                             --model_name $new_name"_classChoice" \
#                             --fields_file ./ids_fields/geeksec/ids_fields_selected15.txt
#            echo "********当前时间：$current_time 当前算法名称：$name---当前seqdim：$seq_dim-----当前模式：test*******"
#            python main.py  --$name \
#                            --test \
#                            --selected \
#                            --data_dir /data/dataset/process_data_geeksec/selected15/npy \
#                            --class_choice Menti,Murlo,Neris,Osx-trojan,RBot,Smokebot,Sogou,Weasel,benign,convagent,covenant,ddosfamily-tcp-thc-ssl,dos-hulk,dos-slowhttptest,emotet,malware-dridex,malware-ursnif,poshc2,qakbot,thefatrat \
#                            --scale scale \
#                            --seq_dim $seq_dim \
#                            --model_path ./models/geeksec/$new_name \
#                            --model_name $new_name"_classChoice" \
#                            --save_path ./results/geeksec/random_testrandom90$new_name \
#                            --save_name $new_name"_classChoice" \
#                            --fields_file ./ids_fields/geeksec/ids_fields_selected15.txt \
#                            --random_select_class
#    done
#done