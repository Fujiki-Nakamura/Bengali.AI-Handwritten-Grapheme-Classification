#!/bin/bash
logdir="../logs"
expids=("20200203002341" "20200207001939" "20200222091738")
for expid in ${expids[@]}; do
    kaggle datasets init -p "${logdir}/${expid}"
    sed -i -e "s/INSERT_SLUG_HERE/${expid}/g" "${logdir}/${expid}/dataset-metadata.json"
    sed -i -e "s/INSERT_TITLE_HERE/Bengali.AI_${expid}/g" "${logdir}/${expid}/dataset-metadata.json"
    kaggle datasets create -r 'zip' -p "${logdir}/${expid}"
done
