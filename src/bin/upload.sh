#!/bin/bash
# NOTE: upload logdir
logdir="../logs"
upload_d="../logs/tmp/"
trained_d="20200315085210/fold_0/"
name="epoch00184.loss0.053581.bestmet0.996439.pt"
message="${trained_d}/${name}"
mkdir -p "${upload_d}/${trained_d}"
cp "${logdir}/${trained_d}/${name}" "${upload_d}/${trained_d}/${name}"
ls "${upload_d}"
kaggle datasets version --dir-mode 'zip' -p "${upload_d}" -m "${message}"

# # expids=("20200203002341" "20200207001939" "20200222091738")
# expids=("20200309144643")
# for expid in ${expids[@]}; do
#     kaggle datasets init -p "${logdir}/${expid}"
#     sed -i -e "s/INSERT_SLUG_HERE/${expid}/g" "${logdir}/${expid}/dataset-metadata.json"
#     sed -i -e "s/INSERT_TITLE_HERE/Bengali.AI_${expid}/g" "${logdir}/${expid}/dataset-metadata.json"
#     kaggle datasets create -r 'zip' -p "${logdir}/${expid}"
# done
