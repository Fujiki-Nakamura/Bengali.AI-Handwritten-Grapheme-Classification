#!/bin/bash
logdir="../logs"
expid=""
kaggle datasets init -p "${logdir}/${expid}/"
sed -i -e "s/INSERT_SLUG_HERE/${expid}/g" "${logdir}/${expid}/dataset-metadata.json"
sed -i -e "s/INSERT_TITLE_HERE/Bengali.AI_${expid}/g" "${logdir}/${expid}/dataset-metadata.json"
kaggle datasets create -p "${logdir}/${expid}/"
