for i in {0..4}; do
    for j in {0..2}; do
        python scripts/plan.py --dataset hopper-medium-v2 --gpt_loadpath gpt/adacat_$i --seed $j --cdf_act 0.9
    done
done

datasets=("walker2d-medium-v2" "halfcheetah-medium-v2")

for i in {0..4}; do
    for j in {0..2}; do
        for dataset in ${datasets[*]}; do 
            python scripts/plan.py --dataset $dataset --gpt_loadpath gpt/adacat_ema_$i --seed $j
        done
    done
done
