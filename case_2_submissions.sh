# constant_names = ["kjs", "kgj", "kjl", "kgl", "kxg", "kxgi", "kxi", "τ", "η", "kλ", "kμ", "Gprod0"]

attacks=case-2-oref0/data/successful_attacks.json
dag=case-2-oref0/oref0.dot
safe_ranges=case-2-oref0/safe_ranges.json
timesteps_per_intervention=5
confidence=0.1
timesteps=500
baseline_confounders=Gprod0

for i in {0..90}
do
  outfile=case-2-oref0/logs/case-2-oref0/attack-$i.json
  data=case-2-oref0/data/$i.pqt
  bash g_estimation.sh \
  $attacks \
  $dag \
  $safe_ranges \
  $timesteps_per_intervention \
  $confidence \
  $i \
  $outfile \
  $timesteps \
  $baseline_confounders \
  $data
done
