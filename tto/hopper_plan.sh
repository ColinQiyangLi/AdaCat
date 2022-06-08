for i in {0..4}; do
    for j in {0..2}; do
        python scripts/plan.py --dataset hopper-medium-v2 --gpt_loadpath gpt/adacat_$i --seed $j --cdf_act 0.0
    done
done
