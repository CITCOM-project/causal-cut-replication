attacks=case-2-oref0/successful_attacks.json
dag=case-2-oref0/oref0.dot
safe_ranges=case-2-oref0/safe_ranges.json
timesteps_per_intervention=5
confidence=0.1
timesteps=500
baseline_confounders="kjs kgj kjl kgl kxg kxgi kxi τ η kλ kμ Gprod0"
logs=case-2-oref0/logs/demo_test

attack_ids="2"


for i in $attack_ids
do
  outfile=$logs/attack-$i.json
  data=case-2-oref0/data-fuzz/$i.csv
  bash g_estimation.sh \
  $attacks \
  $dag \
  $safe_ranges \
  $timesteps_per_intervention \
  $confidence \
  $i \
  $outfile \
  $timesteps \
  "$baseline_confounders" \
  $data
done

python process_results.py $logs
