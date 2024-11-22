root=case-1-swat
attacks=$root/successful_attacks_mutated.json
dag=$root/dcg_raw.dot
safe_ranges=$root/safe_ranges.json
timesteps_per_intervention=15
timesteps=500
baseline_confounders="time"
data=$root/data/data-210.csv

for confidence in 0.1 0.2 0.3
do
  ci=$(awk "BEGIN { printf \"%.0f\n\", 100 - 100 * $confidence }")
  logs="${root}/logs/data-${ci}"
  for i in {1..119}
  do
    outfile=$logs/attack-${i}.json
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
  wait

  python process_results.py $logs
done
