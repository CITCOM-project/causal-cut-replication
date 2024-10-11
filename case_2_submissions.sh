# constant_names = [kjs kgj kjl kgl kxg kxgi kxi τ η kλ kμ Gprod0]

attacks=../APSDigitalTwin/new_successful_attacks.json
dag=case-2-oref0/oref0.dot
safe_ranges=case-2-oref0/safe_ranges.json
timesteps_per_intervention=5
confidence=0.3
timesteps=500
# baseline_confounders="time"
baseline_confounders="kjs kgj kjl kgl kxg kxgi kxi τ η kλ kμ Gprod0"
logs=case-2-oref0/logs/new_attacks

for i in 2 9 19
do
  outfile=$logs/attack-$i.json
  # data=case-2-oref0/data/$i.csv
  data=../APSDigitalTwin/data-ctrl-trt/chunks/$i.csv
  # data=../APSDigitalTwin/data-1000/chunks/$i.csv
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
