attacks=../APSDigitalTwin/new_successful_attacks.json
dag=case-2-oref0/oref0.dot
safe_ranges=case-2-oref0/safe_ranges.json
timesteps_per_intervention=5
confidence=0.1
timesteps=500
baseline_confounders="kjs kgj kjl kgl kxg kxgi kxi τ η kλ kμ Gprod0"
logs=case-2-oref0/logs/new_attacks_fuzz_90_stomach

attack_ids="2 10 14 16 18 19 35 42 45 56 57 59 68 70 84 85 105 112 116 118 126 129 130 131 137 138 139 145 157 163 164 169 176 177 182 183 195 215 219 220 244 276 280 285 289 292 309 311 333 337 339 361 363 364 379 381 382 390 392 395 401 430 445 448 451 456 460 463 469 477"


threads=8
for i in $attack_ids
do
  ((thread=thread%threads)); ((thread++==0)) && wait
  outfile=$logs/attack-$i.json
  # data=case-2-oref0/data/$i.csv
  data=../APSDigitalTwin/data-fuzz/chunks/$i.csv
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
  $data &
done
wait

python process_results.py $logs
