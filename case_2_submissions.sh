root=case-2-oref0
attacks=$root/successful_attacks.json
dag=$root/dcg.dot
safe_ranges=$root/safe_ranges.json
timesteps_per_intervention=5
timesteps=500
baseline_confounders="kjs kgj kjl kgl kxg kxgi kxi τ η kλ kμ Gprod0"
attack_ids="2 10 14 16 18 19 35 42 45 56 57 59 68 70 84 85 105 112 116 118 126 129 130 131 137 138 139 145 157 163 164 169 176 177 182 183 195 215 219 220 244 276 280 285 289 292 309 311 333 337 339 361 363 364 379 381 382 390 392 395 401 430 445 448 451 456 460 463 469 477"

threads=8

for confidence in "0.1, 0.2, 0.3"
do
  ci=$(awk "BEGIN { printf \"%.0f\n\", 100 - 100 * $x }")
  for dataset in {0..9}
  do
    logs="${root}/logs/${dataset}-${ci}"
    for i in $attack_ids
    do
      ((thread=thread%threads)); ((thread++==0)) && wait
      outfile=$logs/attack-$i.json
      data=fuzz_data_$dataset/$i.csv
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
  done
done
